import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from functions.run_handler import Run_Handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def main():
    try:
        run_handler = Run_Handler()
        run_handler.run()
    except KeyboardInterrupt:
        logger.info("Processus interrompu par l'utilisateur.")
    except Exception as e:
        logger.error(f"Une erreur est survenue: {e}")


if __name__ == "__main__":
    main()
