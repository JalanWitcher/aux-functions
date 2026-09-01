from scipy.integrate import OdeSolution
import numpy as np

class OdeResultPassos:
    from scipy.integrate import OdeSolution
    import numpy as np
    
    def __init__(self, events = []):
        self.success = True
        self.t = np.array([])
        self.sol = None
        self.t_min = 1
        self.t_max = -1
        self.Eventos = {event.__name__ : np.array([]) for event in events}
        self.events = events
        self.Integrou = True
        self.UnicoInterp = True
        self.lastT = -np.inf
        self.tTranslac = 0

    def __call__(self, t):
        return self.sol(np.asarray(t))

    def joinInterpolador(self, interpolador, updateLast=True):
        if self.UnicoInterp:
            if self.sol is None:
                self.sol = interpolador
                self.t_min = interpolador.t_min
                self.t_max = interpolador.t_max
            else:
                interps = self.sol.interpolants + interpolador.interpolants
                ts = np.concatenate((self.sol.ts[:-1], interpolador.ts))
                self.sol = self.OdeSolution(ts, interps)
                self.t_min = np.min(ts)
                self.t_max = np.max(ts)
        else:
            if updateLast:
                self.sol.updateInterpolador(interpolador)
            else:
                self.sol.joinInterpolador(interpolador)
            self.t_min = self.sol.t_min
            self.t_max = self.sol.t_max

    def concatenaSolucao(self, Sol, Integrou=True):
        self.success *= Sol.success
        self.Integrou = self.Integrou and Integrou

        if Sol.t[0] < self.lastT:
            updateLast = False
            self.tTranslac = self.t_max
            if self.UnicoInterp:
                self.sol = self.SolucaoConjunta(self.sol)
                self.UnicoInterp = False
        else:
            updateLast = True
        self.t = np.concatenate((self.t[:-1], Sol.t + self.tTranslac))


        for j, event in enumerate(self.events):
            key = event.__name__
            self.Eventos[key] = np.concatenate((self.Eventos[key], Sol.t_events[j]+self.tTranslac))

        self.lastT = Sol.t[-1]
        self.joinInterpolador(Sol.sol, updateLast=updateLast)

    @property
    def y(self): return self.sol(np.asarray(self.t))

    @property
    def t_events(self) -> list: return list(self.Eventos.values())

    @property
    def y_events(self): return [self.sol(tEvent) for tEvent in self.t_events]

    def __repr__(self):
        strRepr = ''
        with np.printoptions(precision=3, edgeitems=3, threshold=10):
            for key in self.__dict__:
                strRepr += f"{key:>10}: {self.__dict__[key].__str__():<50}\n"
                # print(f"{key:>10}: {self.__dict__[key].__str__():<50}")
        return strRepr

    class SolucaoConjunta:
        from scipy.integrate import OdeSolution

        def __init__(self, Interpoladores):
            if not isinstance(Interpoladores, list):
                Interpoladores = [Interpoladores]
            self.Interpoladores = Interpoladores
            self.tSol = np.asarray([interp.t_max - interp.t_min for interp in self.Interpoladores])
            self.updateCumTime()

        def __call__(self, t):
            tLocal = np.atleast_1d(t)

            Y_shape = self.Interpoladores[0](0).size
            Y = np.empty((Y_shape, tLocal.size))

            for tStart, tEnd, interpolador in zip(self.cum_t[:-1], self.cum_t[1:], self.Interpoladores):
                index_inter = (tLocal >= tStart) * (tLocal <= tEnd)
                if np.any(index_inter):
                    Y[:, index_inter] = interpolador(tLocal[index_inter] - tStart*(tStart > 0))

            return Y if np.asarray(t).ndim > 0 else Y[:, 0]

        def updateCumTime(self):
            self.cum_t = np.concatenate(([0], np.cumsum(self.tSol)))
            self.t_min = self.cum_t[0]
            self.t_max = self.cum_t[-1]

        def joinInterpolador(self, interpolador):
            self.Interpoladores += [interpolador]
            self.tSol = np.concatenate((self.tSol, [interpolador.t_max - interpolador.t_min]))
            self.updateCumTime()

        def updateInterpolador(self, interpolador):
            lastInterp = self.Interpoladores[-1]
            interps = lastInterp.interpolants + interpolador.interpolants
            ts = np.concatenate((lastInterp.ts[:-1], interpolador.ts))
            self.Interpoladores[-1] = self.OdeSolution(ts, interps)
            self.tSol[-1] = self.Interpoladores[-1].t_max - self.Interpoladores[-1].t_min
            self.updateCumTime()

def integracaoIntervalos(funIVP, tIntervalos, Y0, events=None, SolOld=None, **options):
    from scipy.integrate import solve_ivp

    if SolOld is None:
        SolFinal = OdeResultPassos(events=events)
    else:
        SolFinal = SolOld

    Y0Atu = Y0
    nEstados = len(funIVP)
    for i, (t1,t2) in enumerate(zip(tIntervalos[:-1], tIntervalos[1:])):
        Sol = solve_ivp(fun=funIVP[i%nEstados], t_span=(t1,t2), dense_output=True, y0=Y0Atu, events=events, **options)
        # SolFinal.concatenaSolucao(Sol)
        Y0Atu = Sol.y[:,-1]
        if Sol.status == 1:
            SolFinal.concatenaSolucao(Sol, Integrou=False)
            break
        SolFinal.concatenaSolucao(Sol, Integrou=True)

    return SolFinal