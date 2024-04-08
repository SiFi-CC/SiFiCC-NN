import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma, A, c):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu) / sigma) ** 2) + c


def lorentzian(x, mu, sigma, A, c):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2) + c


def lorentzian_pol2(x, mu, sigma, A, c, p4, p5):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2) + c\
           + x * p4 + x ** 2 * p5


def gaussian_pol2(x, mu, sigma, A, c, p4, p5):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu) / sigma) ** 2) + c + x * p4 + x ** 2 * p5


def gaussian_lorentzian(x, mu1, sigma1, A1, mu2, sigma2, A2, c):
    return c +\
           A1 / np.pi * (1 / 2 * sigma1) / ((x - mu1) ** 2 + (1 / 2 * sigma1) ** 2) +\
           A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu2) / sigma2) ** 2)


def gaussian_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2, c):
    return c + A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu1) / sigma1) ** 2) + A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu2) / sigma2) ** 2)


def auto_hist_fitting(f,
                      bins,
                      hist,
                      p0=None):
    # list of supported auto-fitting functions
    dict_f = {"gaussian": gaussian,
              "lorentzian": lorentzian,
              "gaussian_pol2": gaussian_pol2,
              "gaussian_gaussian": gaussian_gaussian,
              "gaussian_lorentzian": gaussian_lorentzian,
              "lorentzian_pol2": lorentzian_pol2}

    # prepare fitting input and remove histogram entries with 0, ruins fitting
    xdata = bins[hist != 0]
    ydata = hist[hist != 0]
    sigma = np.sqrt(ydata)
    width = bins[1] - bins[0]
    x = np.linspace(min(bins), max(bins), 1000)

    if f in ["gaussian", "lorentzian"]:
        popt, pcov = curve_fit(f=dict_f[f],
                               xdata=xdata,
                               ydata=ydata,
                               sigma=sigma,
                               p0=p0 if p0 is not None else [0.0, 1.0, np.sum(hist) * width, 0])
        fx = dict_f[f](x, *popt)
        return popt, pcov, x, fx

    elif f == "gaussian_pol2":
        popt, pcov = curve_fit(f=dict_f[f],
                               xdata=xdata,
                               ydata=ydata,
                               sigma=sigma,
                               p0=p0 if p0 is not None else [0.0, 1.0, np.sum(hist) * width,
                                                             0, 0.015, 0.06],
                               maxfev=2000)
        fx = dict_f[f](x, *popt)
        return popt, pcov, x, fx

    elif f == "lorentzian_pol2":
        popt, pcov = curve_fit(f=dict_f[f],
                               xdata=xdata,
                               ydata=ydata,
                               sigma=sigma,
                               p0=p0 if p0 is not None else [0.0, 1.0, np.sum(hist) * width,
                                                             0, 0.015, 0.06],
                               maxfev=2000)
        fx = dict_f[f](x, *popt)
        return popt, pcov, x, fx

    elif f == "gaussian_lorentzian":
        popt, pcov = curve_fit(f=dict_f[f],
                               xdata=xdata,
                               ydata=ydata,
                               sigma=sigma,
                               p0=p0 if p0 is not None else [0.0, 1.0, np.sum(hist) * width,
                                                             0.0, 0.5, np.sum(hist) * width, 0],
                               maxfev=2000)
        fx = dict_f[f](x, *popt)
        return popt, pcov, x, fx

    elif f == "gaussian_gaussian":
        popt, pcov = curve_fit(f=dict_f[f],
                               xdata=xdata,
                               ydata=ydata,
                               sigma=sigma,
                               p0=p0 if p0 is not None else [0.0, 1.0, np.sum(hist) * width,
                                                             0.0, 0.5, np.sum(hist) * width, 0])
        fx = dict_f[f](x, *popt)
        return popt, pcov, x, fx
