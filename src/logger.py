import logging
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

class Logger:
  def __init__(self, log_name: str, log_dir="logs" ):
    
    log_dir_path = ROOT_DIR / log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"{log_name}.log"
    self.logger = logging.getLogger(log_name)
    self.logger.setLevel(logging.INFO)
    
    if not self.logger.handlers:
      handler = logging.FileHandler(log_path, mode='a')
      handler.setFormatter(logging.Formatter(
					"%(asctime)s - %(levelname)s - %(message)s"
			))
      self.logger.addHandler(handler)
    
    
  def info(self, message, *args, **kwargs):
      self.logger.info(message, *args, **kwargs)

  def warning(self, message, *args, **kwargs):
      self.logger.warning(message, *args, **kwargs)

  def error(self, message, *args, **kwargs):
      self.logger.error(message, *args, **kwargs)

  def debug(self, message, *args, **kwargs):
      self.logger.debug(message, *args, **kwargs)
    
		
