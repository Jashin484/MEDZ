import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return round(x / self.discrete)



class RickerPlaneWave:
    ''' Класс с уравнением плоской волны для гармонического сигнала в дискретном виде
    Np - коэффициент риккера.
    Md - коэффициент риккера.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self,Np = 30, Md = 2.5, Sc=1.0, eps=1.0, mu=1.0):
        self.Np = Np
        self.Md = Md
        self.eps = eps
        self.Sc = Sc/np.sqrt(self.eps)
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return ((1 - 2 * numpy.pi ** 2 * ((self.Sc * (q -m)) / self.Np - self.Md) ** 2) *
                          numpy.exp(-numpy.pi ** 2 * ((self.Sc * (q - m)) / self.Np - self.Md) ** 2))


if __name__ == '__main__':

    c = 3e8

    # Волновое сопротивление свободного пространства
    eps_r = 1.5
    Z0 = 120.0 * numpy.pi

    # Расчет длинны волны через частоты сигнал
    f_min = 0.1e9
    f_max = 2e9
    f_mid = (f_max+f_min)/2
    lamda = c / f_mid

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в отсчетах
    maxSize_m = 4.5
    dx = 5e-3
    dt = dx/c
    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)



    # Время расчета в отсчетах
    maxTime_s = 100e-9
    maxTime = sampler_t.sample(maxTime_s)
    # Параметры Вейвлета Рикера
    Np = 55.0
    Md = 2.5

    # Датчики для регистрации поля
    sourcePos = maxSize // 3
    probesPos = [sourcePos + int(0.33 * maxSize)]
    probes = [Probe(pos, maxTime) for pos in probesPos]


    # Где начинается PEC
    PEC_x = maxSize - int(0.1 * maxSize)
    # Диэлектрическая проницаемость

    eps = numpy.ones(maxSize)
    eps[:] = eps_r

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = RickerPlaneWave (Np, Md, Sc, eps[sourcePos], mu[sourcePos])
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    pec = numpy.zeros(maxSize)
    pec_m = numpy.zeros(maxSize - 1)
    pec[PEC_x] = 1
    pec_m[:] = pec[:-1]

     # Коэффициенты для расчета поля E
    ceze = (1.0 - pec) / (1.0 + pec)
    cezh = (Sc * Z0) / (eps * (1.0 + pec))

    # Коэффициенты для расчета поля H
    chyh = (1.0 - pec_m) / (1.0 + pec_m)
    chye = Sc / (mu * Z0 * (1.0 + pec_m))

    oldEzLeft = Ez[1]
    Sc1Left = Sc / numpy.sqrt(mu[0] * eps[0])
    koeffABCLeft = (Sc1Left - 1) / (Sc1Left + 1)

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = AnimateFieldDisplay(dx, dt, maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy[:] = chyh * Hy + chye * (Ez[1:] - Ez[:-1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1: -1] * Ez[1:-1] + cezh[1: -1] * (Hy[1:] - Hy[:-1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 10 == 0:
            display.updateData(display_field, q)
    display.stop()

    showProbeSpectrum(probes, dx, dt, -1.1, 1.1)