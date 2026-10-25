"""
swi_llm/agent/sandbox.py

execute_python ツールのサンドボックス実行 (REQ-01 で executor.py から分離)。

Python 実行のサンドボックスは次の 2 段構え:
  1. Docker コンテナ（カーネルレベル分離、ネットワーク遮断、メモリ制限）
  2. Docker 不可時の AST ホワイトリスト検査 + サブプロセス実行（フォールバック）

フォールバックの AST 検査はホワイトリスト方式 (REQ-20):
許可モジュール (SANDBOX_ALLOWED_MODULES) 以外の import、危険な組み込み関数、
アンダースコア始まりの属性アクセス (REQ-21: `__class__` 等の dunder 経由の
エスケープ防止) を全て拒否する。
"""
import os
import ast
import base64
import glob
import subprocess
import sys
import tempfile

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    PYTHON_EXEC_TIMEOUT_SEC,
    DOCKER_SANDBOX_IMAGE, DOCKER_SANDBOX_MEM_LIMIT,
    DOCKER_SANDBOX_USER, DOCKER_SANDBOX_PIDS_LIMIT, DOCKER_SANDBOX_NANO_CPUS,
    DOCKER_SANDBOX_CAP_DROP, DOCKER_SANDBOX_SECURITY_OPT,
    SANDBOX_ALLOWED_MODULES, SANDBOX_FORBIDDEN_BUILTINS,
)

logger = setup_logger("agent.sandbox")


def _validate_code_ast(code):
    """
    AST フォールバック・サンドボックスの静的検査（ホワイトリスト方式）。

    拒否事由が見つかった場合はその内容の文字列を返し、問題なければ None を返す。
      - 許可リスト (SANDBOX_ALLOWED_MODULES) 外モジュールの import (REQ-20)
      - 危険な組み込み関数の呼び出し・参照 (REQ-20)
      - アンダースコア始まりの属性アクセス (`__class__` 等の dunder 探索による
        サンドボックスエスケープの防止) (REQ-21)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in SANDBOX_ALLOWED_MODULES:
                    return f"Security Error: Importing '{alias.name}' is not allowed (whitelist policy)."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split('.')[0]
            if root not in SANDBOX_ALLOWED_MODULES:
                return f"Security Error: Importing from '{node.module}' is not allowed (whitelist policy)."
        elif isinstance(node, ast.Attribute):
            # __class__ / __globals__ / _private 等、内部属性への到達を遮断
            if node.attr.startswith("_"):
                return f"Security Error: Access to attribute '{node.attr}' is forbidden."
        elif isinstance(node, ast.Name):
            if node.id in SANDBOX_FORBIDDEN_BUILTINS or node.id.startswith("__"):
                return f"Security Error: Use of '{node.id}' is forbidden."

    return None


def execute_python(code):
    """
    提供されたPythonコードをDockerコンテナ（サンドボックス）で実行し、
    標準出力、エラー出力、および生成された画像（Base64）を返します。
    Dockerが利用できない場合はASTホワイトリスト検査+サブプロセス実行に
    フォールバックします。
    """
    try:
        import docker
        client = docker.from_env()
        use_docker = True
    except Exception as e:
        # フォールバック発生をログで検知できるようにする (REQ-22)
        logger.warning(f"Docker unavailable ({e}). Falling back to AST-whitelist sandbox.")
        use_docker = False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            if use_docker:
                try:
                    # SEC-5: 非rootユーザ(nobody)で /app（rwバインドマウント）を読み書きできるよう、
                    # ホスト側の一時ディレクトリを全許可にする。これが無いと native Linux では
                    # コンテナのuid(65534)が script.py を読めず/生成物を書けず実行が壊れる。
                    os.chmod(temp_dir, 0o777)
                    # Dockerを使用した真のカーネルレベルサンドボックス。
                    # SEC-5: network遮断/メモリ制限に加え、多層防御として非root化・全ケーパビリティ
                    # 剥奪・権限昇格禁止・PID/CPU制限・rootfs読取専用を課す（脱出/資源枯渇の被害最小化）。
                    # HOME/MPLCONFIGDIR を rwマウントの /app に向け、read_only下でも matplotlib 等の
                    # キャッシュ書込（生成画像機能）が壊れないようにする。
                    container = client.containers.run(
                        DOCKER_SANDBOX_IMAGE,
                        command=["python", "/app/script.py"],
                        volumes={temp_dir: {'bind': '/app', 'mode': 'rw'}},
                        working_dir="/app",
                        mem_limit=DOCKER_SANDBOX_MEM_LIMIT,
                        network_disabled=True,
                        detach=True,
                        user=DOCKER_SANDBOX_USER,
                        cap_drop=list(DOCKER_SANDBOX_CAP_DROP),
                        security_opt=list(DOCKER_SANDBOX_SECURITY_OPT),
                        pids_limit=DOCKER_SANDBOX_PIDS_LIMIT,
                        nano_cpus=DOCKER_SANDBOX_NANO_CPUS,
                        read_only=True,
                        environment={"HOME": "/app", "MPLCONFIGDIR": "/app"},
                    )

                    try:
                        result = container.wait(timeout=PYTHON_EXEC_TIMEOUT_SEC)
                        output = container.logs(stdout=True, stderr=False).decode('utf-8')
                        error = container.logs(stdout=False, stderr=True).decode('utf-8')
                        returncode = result['StatusCode']
                    except Exception as e:
                        container.stop()
                        logger.error(f"Docker execution error or timeout: {e}")
                        return {"status": "error", "error_message": f"Docker Execution Error or Timeout: {e}"}
                    finally:
                        container.remove(force=True)

                except Exception as e:
                    logger.error(f"Docker initialization failed: {e}")
                    return {"status": "error", "error_message": f"Docker initialization failed: {e}"}
            else:
                # SEC-1: Docker が使えない環境ではカーネルレベルの隔離が無く、AST 静的検査だけでは
                # operator.attrgetter / str.format 等の間接アクセスで脱出しうる（PoC 実証済み）。
                # 加えてフォールバック subprocess はネットワーク/メモリ/CPU 非隔離のため、脱出は
                # 即ホスト RCE となる。よって「隔離なし実行」は既定で拒否し、リスクを明示的に受容する
                # 場合のみ SWI_ALLOW_UNSANDBOXED_EXEC=1 でオプトインさせる。
                if os.environ.get("SWI_ALLOW_UNSANDBOXED_EXEC", "").strip().lower() not in ("1", "true", "yes"):
                    logger.warning("execute_python refused: Docker sandbox unavailable and unsandboxed "
                                   "fallback is disabled by default (set SWI_ALLOW_UNSANDBOXED_EXEC=1 to opt in).")
                    return {"status": "error",
                            "error_message": ("コード実行サンドボックス (Docker) が利用できないため、"
                                              "安全のためコード実行を拒否しました。")}

                # フォールバック: ASTホワイトリスト検査+サブプロセス実行（オプトイン時のみ）
                rejection = _validate_code_ast(code)
                if rejection is not None:
                    # サンドボックス拒否イベントを記録 (REQ-08)
                    logger.warning(f"Sandbox rejected code: {rejection} | code={code[:200]!r}")
                    return {"status": "error", "error_message": rejection}

                try:
                    result = subprocess.run(
                        [sys.executable, "-I", "script.py"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=PYTHON_EXEC_TIMEOUT_SEC,
                        encoding='utf-8'
                    )
                    output = result.stdout
                    error = result.stderr
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    logger.warning(f"Sandboxed code timed out after {PYTHON_EXEC_TIMEOUT_SEC}s.")
                    return {"status": "error", "error_message": f"Execution timed out ({PYTHON_EXEC_TIMEOUT_SEC} seconds limit)."}

            # 生成された画像ファイルの回収
            images_base64 = []
            for img_file in glob.glob(os.path.join(temp_dir, "*.png")) + glob.glob(os.path.join(temp_dir, "*.jpg")):
                with open(img_file, "rb") as img_f:
                    encoded = base64.b64encode(img_f.read()).decode('utf-8')
                    mime_type = "image/png" if img_file.endswith(".png") else "image/jpeg"
                    images_base64.append(f"data:{mime_type};base64,{encoded}")

            response = {
                "status": "success" if returncode == 0 else "error",
                "output": output.strip() if output else "No output"
            }
            if returncode != 0:
                response["error_message"] = error.strip() if error else "Unknown error"
                logger.error(f"execute_python finished with returncode={returncode}: {response['error_message'][:300]}")
            else:
                logger.info(f"execute_python succeeded. output={response['output'][:200]!r}")
            if images_base64:
                response["images"] = images_base64

            return response

    except Exception as e:
        logger.exception("execute_python failed unexpectedly.")
        return {"status": "error", "error_message": str(e)}
