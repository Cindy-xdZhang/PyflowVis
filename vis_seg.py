from FLowUtils.netCDFLoader import *


def test_netcdf(url):
    ret=NetCDFLoader.load_vector_field2d(url) 


if __name__ == '__main__':
    test_netcdf("C:\\Users\\zhanx0o\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowCDFdata\\cylinder2d.nc")
