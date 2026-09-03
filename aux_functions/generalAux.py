

def identificaNumeros(string, dtype=float):
    import re
    import numpy as np
    return np.array(re.findall(r"[-+]?(?:\d*\.*\d+)", string), dtype=dtype)

def identificaLetras(string, pattern=r'[a-z]+'):
    import re
    return re.findall(pattern, string)

def achaCasasDecimais(xArray):
    if len(xArray) < 0:
      return

    from decimal import Decimal
    import numpy as np

    diffX = np.round(np.diff(xArray), 10)
    stepX = np.min(np.abs(diffX[np.nonzero(diffX)]))
    decimalsX = abs(Decimal(str(stepX)).as_tuple().exponent)

    return decimalsX

def isEven(n):
    return n % 2 == 0