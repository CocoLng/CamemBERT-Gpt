import logging

from functions.run_handler import Run_Handler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Désactiver les logs du module datasets en dessous du niveau WARNING
logging.getLogger("datasets_modules.datasets").setLevel(logging.WARNING)


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
