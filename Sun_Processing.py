import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting

x_dimension = 5202
y_dimension = 3464
master_dark = np.zeros((y_dimension, x_dimension), dtype = float)
rgb_index = [0.3, 0.59, 0.11]

def greyscale(fits_file_data):
    return np.dot(fits_file_data.T, rgb_index).T

def names(str_0, n):
    name_list = []
    for i in range(n):
        name_list.append(str_0 + ("0000" + str(i + 1))[-5:] + ".fits")
    return name_list

def sigmaclip(names, n, master_dark):
    number_of_images = len(names)
    total_data = np.zeros((number_of_images, y_dimension, x_dimension), dtype = float)
    for i in range(number_of_images):
        fits_file = fits.open(get_pkg_data_filename(names[i]))
        fits_file_data = fits_file[0].data
        total_data[i] = greyscale(fits_file_data)
        fits_file.close()
        print("Images Stacked: ", i + 1)
    total_data = total_data.T
    for i in range(len(total_data)):
        for j in range(len(total_data[i])):
            median = np.median(total_data[i][j])
            sigma = np.std(total_data[i][j])
            for k in total_data[i][j]:
                if abs(k - median) >= n * sigma:
                    k = np.nan
            master_dark[j][i] = np.nanmean(total_data[i][j])
        if i % 100 == 0:
            print("Processing : ", round(((i / 5202) * 100), 2), "%")
    print("Processing : ", 100, "%")
            
def Histogram_image(Master_data, title, xlabel, ylabel, bins):
    print("Master data median: ", np.median(Master_data))
    print("Master data mean", np.mean(Master_data))
    print("Master data standard deviation", np.std(Master_data))
    counts, bins = np.histogram(Master_data, bins)
    plt.stairs(counts, bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, 600])
    plt.show()
    
    
name_list = names("Canon_1200D_Dark_30_ISO_100_", 20)
sigmaclip(name_list, 3, master_dark)
Histogram_image(master_dark, "Dark Histogram", "count", "freq", 7000)

x_centre = 2984
y_centre = 2279
radius = 794

sun_file = get_pkg_data_filename("Sun_00001.fits")
fits.info(sun_file)
sun_data = fits.getdata(sun_file, ext = 0)
grey_sun_data = greyscale(sun_data)
dark_corrected_sun_data = grey_sun_data - master_dark

def mu_list(radius, n):
    mu_list = []
    centre_pixel = 0
    for i in range(int(radius / n)):
        mu_list.append(np.sqrt(1 - (centre_pixel / radius) ** 2))
        centre_pixel += n
    return mu_list

def surface_brightness(dark_corrected_sun_data, n, radius):
    surface_brightness_list = np.zeros(int(radius / n), dtype = float)
    unc_list = np.zeros(int(radius / n), dtype = float)
    centre_pixel = 0
    for k in range(int(radius / n)):
        total_brightness = 0
        box = np.zeros((n, n), dtype = float)
        for i in range(n):
            for j in range(n):
                total_brightness += dark_corrected_sun_data[y_dimension - (y_centre + i - n // 2)][x_centre + centre_pixel + j - n // 2]
                box[i][j]  = dark_corrected_sun_data[y_dimension - (y_centre + i - n // 2)][x_centre + centre_pixel + j - n // 2]
        surface_brightness_list[k] = total_brightness
        unc_list[k] = np.std(box)
        centre_pixel += n
    max_brightness = max(surface_brightness_list)
    for m in range(int(radius / n)):
        surface_brightness_list[m] = surface_brightness_list[m] / max_brightness
        unc_list[m] = unc_list[m] / max_brightness
    return surface_brightness_list, unc_list

x = mu_list(radius, 9)
y, yunc = surface_brightness(dark_corrected_sun_data, 9, radius)
line_orig = models.Linear1D(slope = 0.6, intercept = 0.4)
line_init = models.Linear1D()
fit = fitting.LinearLSQFitter()
fitted_line = fit(line_init, x, y, weights = 1.0 / (yunc ** 2))
plt.errorbar(x, y, yerr = yunc, fmt = 'ko', label = "Data")
plt.plot(x, fitted_line(x), label = "Fitted Model")
plt.plot(x, line_orig(x), label = "Theoretical Model")
plt.xlabel("mu")
plt.ylabel("surface brightness")    
plt.title("Solar Intensity Profile")
plt.legend()
plt.show()

plt.figure()
pixel = [9 * i for i in range(200)]
y_prime, yunc_prime = surface_brightness(dark_corrected_sun_data, 9, 1800)
plt.plot(pixel, y_prime, '.')
plt.xlabel("Distance from the centre in pixels")
plt.ylabel("surface_brightness")
plt.title("Solar Intensity Profile")
plt.show()