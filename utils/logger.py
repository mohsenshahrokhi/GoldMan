import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('goldman.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('GoldMan')


logger = setup_logger()
