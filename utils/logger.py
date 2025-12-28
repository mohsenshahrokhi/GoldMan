import logging
import time
from collections import defaultdict


class HTTPRequestFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self._last_log_time = defaultdict(float)
        self._log_interval = 300
    
    def filter(self, record):
        if record.name == 'httpx' and 'HTTP Request' in record.getMessage():
            current_time = time.time()
            last_time = self._last_log_time.get('httpx_requests', 0)
            
            if current_time - last_time >= self._log_interval:
                self._last_log_time['httpx_requests'] = current_time
                return True
            return False
        return True


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('goldman.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    http_filter = HTTPRequestFilter()
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.addFilter(http_filter)
    
    return logging.getLogger('GoldMan')


logger = setup_logger()
