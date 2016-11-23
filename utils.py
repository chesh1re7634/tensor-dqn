from datetime import datetime

def tmpLogDir(base_path='/tmp/log_'):
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    return base_path + suffix