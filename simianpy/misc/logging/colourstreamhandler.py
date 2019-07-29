import logging

from colorama import Fore, Back, Style

class ColourStreamHandler(logging.StreamHandler):
    """Colour stream handler utility class

    A stream handler that uses colorama to provide colour output in terminal windows.

    Parameters
    ----------
    colours: dict, optional
        colours to override defaults
    **kwargs: optional
        Optional keyword arguments passed to parent (logging.StreamHandler)

    Attributes
    ----------
    default_colours: dict
        The default colours used by the ColourStreamHandler - can be overridden on init
    """
    default_colours = {
            'DEBUG':Fore.CYAN,
            'INFO':Fore.GREEN,
            'WARN':Fore.YELLOW,
            'WARNING':Back.BLUE + Fore.YELLOW,
            'ERROR': Back.RED + Fore.WHITE,
            'CRIT': Back.RED + Fore.WHITE,
            'CRITICAL': Back.RED + Fore.WHITE
        }
    def __init__(self, colours = {}, **kwargs):
        for k,v in self.default_colours.items():
            if k not in colours:
                colours[k] = v
        self.colours = colours
        super().__init__(**kwargs)

    def emit(self, record):
        try:
            message = self.format(record)
            self.stream.write(self.colours[record.levelname] + message + Style.RESET_ALL)
            self.stream.write(getattr(self, 'terminator', '\n'))
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)