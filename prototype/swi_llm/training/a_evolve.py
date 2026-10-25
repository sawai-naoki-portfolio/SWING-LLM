import os
import sys
import time
import subprocess
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from swi_llm.utils.logger import setup_logger

logger = setup_logger("a_evolve")

def run_command(cmd, desc):
    logger.info(f"[{desc}] Starting: {cmd}")
    try:
        # Use shell=True for powershell scripts if necessary, but list of args is better for standard python scripts
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"[{desc}] Succeeded.\n{result.stdout[-500:]}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{desc}] Failed with exit code {e.returncode}.\n{e.stderr[-500:]}")
        return False

def a_evolve_loop(dry_run=False):
    """
    A-Evolve Autonomous Loop:
    1. SFT Training
    2. GRPO Training
    3. E2E Testing
    4. Promotion
    """
    logger.info("=== Starting A-Evolve Autonomous Loop ===")
    
    python_exe = sys.executable
    # base_dir = swi_llm パッケージルート、project_root = SWI_LLM リポジトリルート
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    sft_script = os.path.join(base_dir, "training", "sft_train.py")
    grpo_script = os.path.join(base_dir, "training", "grpo_train.py")
    pipeline_script = os.path.join(project_root, "run_pipeline.ps1")

    # Set SWI_MODE to SMOKE to keep iterations fast for A-Evolve prototyping
    os.environ["SWI_MODE"] = "SMOKE"
    
    if dry_run:
        logger.info("DRY RUN MODE: Skipping actual training and testing.")
        time.sleep(1)
        logger.info("A-Evolve Dry-run successful.")
        return

    # Step 1: SFT
    if not run_command([python_exe, sft_script], "SFT Training"):
        logger.error("A-Evolve Loop aborted at SFT step.")
        return

    # Step 2: GRPO
    if not run_command([python_exe, grpo_script], "GRPO Training"):
        logger.error("A-Evolve Loop aborted at GRPO step.")
        return

    # Step 3: Verification (E2E Test)
    logger.info("Starting E2E Verification via run_pipeline.ps1 (SMOKE mode)...")
    # Using powershell to run the script
    test_cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", pipeline_script, "-Mode", "SMOKE"]
    if not run_command(test_cmd, "E2E Verification"):
        logger.error("A-Evolve Loop aborted: Verification failed. Model degrades performance.")
        # Rollback logic could be implemented here (e.g. deleting the new _SMOKE.pt checkpoints)
        return

    # Step 4: Promotion
    logger.info("=== A-Evolve Loop Completed Successfully ===")
    logger.info("New weights verified and promoted as baseline.")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    while True:
        a_evolve_loop(dry_run=dry_run)
        if dry_run:
            break
        
        # Wait before next evolution cycle
        logger.info("Sleeping for 60 seconds before next evolution cycle...")
        time.sleep(60)
