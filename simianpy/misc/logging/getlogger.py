import logging

def getLogger(loggerName, fileName = '', fileMode = 'a', fileLevel = 'DEBUG', printLevel = 'WARN', colours = {}):
    """Utility that returns a logger object

    Parameters
    ----------
    loggerName: str
        Name for the logger, usually just provide __name__
    fileName: str or bool, optional
        Name for the log file, if False no log file is generated
    fileMode: str, optional, default: 'a'
        Mode for opening log file (e.g. 'w' for write, 'a' for append).
    fileLevel: {'DEBUG','INFO','WARN'}, optional, default: 'DEBUG'
        Log level for FileHandler
    printLevel: {'DEBUG','INFO','WARN'}, optional, default: 'WARN'
        Log level for StreamHandler
    colours: dict, optional, default: {}
        Override default colours for ColourStreamHandler (see .logging.colourstreamhandler.ColourStreamHandler for more info)

    Returns
    -------
    logger: logging.Logger
        A logger object with FileHandler and StreamHandler as configured. Note the logger is a singleton.
    """
    logger_levels = {'DEBUG':logging.DEBUG, 'INFO':logging.INFO, 'WARN':logging.WARN}

    logger = logging.getLogger(loggerName)
    if logger.hasHandlers():
        logger.handlers[:] = []

    logger.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if fileName:
        assert fileLevel in logger_levels, f"Provided invalid fileLevel '{fileLevel}'. Valid options: {logger_levels.keys()}"
        # create and add file handler
        fileHandler = logging.FileHandler(fileName, mode = fileMode)
        fileHandler.setLevel(logger_levels[fileLevel])
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    assert printLevel in logger_levels, f"Provided invalid printLevel '{printLevel}'. Valid options: {logger_levels.keys()}"
    
    # create and add stream handler
    try:
        import colorama
    except ImportError:
        streamHandler = logging.StreamHandler()
        colour_support = False
    else:
        from .colourstreamhandler import ColourStreamHandler
        colorama.init(autoreset=True)
        streamHandler = ColourStreamHandler(colours)
        colour_support = True
        
    streamHandler.setLevel(printLevel)
    logger.addHandler(streamHandler)

    if colour_support:
        logger.info('Logger initialized with colour support.')
    else:
        logger.info('Logger initialized without colour support. Failed to import colorama.')

    return logger