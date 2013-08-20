# Licensed under a 3-clause BSD style license - see PYFITS.rst

import platform

import numpy as np

from . import FitsTestCase
from ....tests.helper import pytest
from ....io import fits


class TestUintFunctions(FitsTestCase):
    @classmethod
    def setup_class(cls):
        cls.utypes = ('u2', 'u4', 'u8')
        cls.utype_map = {'u2': np.uint16, 'u4': np.uint32, 'u8': np.uint64}
        cls.itype_map = {'u2': np.int16, 'u4': np.int32, 'u8': np.int64}
        cls.format_map = {'u2': 'I', 'u4': 'J', 'u8': 'K'}

    test_uint_parameters = [
        ('u2', False), ('u4', False), ('u8', False), ('u2', True),
        ('u4', True)  # , ('u8', True)
        # Note: CFITSIO doesn't want to compress 64-bit ints
    ]

    @pytest.mark.parametrize(('utype','compressed'), test_uint_parameters)
    def test_uint(self, utype, compressed):
        bits = 8 * int(utype[1])

        if bits == 64 and platform.architecture()[0] != '64bit':
            pytest.skip('Unsupported on 32-bit architecture')

        if compressed:
            hdu = fits.CompImageHDU(np.array([-3, -2, -1, 0, 1, 2, 3]))
            hdu_number = 1
        else:
            hdu = fits.PrimaryHDU(np.array([-3, -2, -1, 0, 1, 2, 3]))
            hdu_number = 0

        hdu.scale('int{0:d}'.format(bits), '', bzero=2 ** (bits-1))
        hdu.writeto(self.temp('tempfile.fits'))

        with fits.open(self.temp('tempfile.fits'), uint=True) as hdul:
            assert hdul[hdu_number].data.dtype == self.utype_map[utype]
            assert (hdul[hdu_number].data ==
                    np.array([(2 ** bits) - 3, (2 ** bits) - 2,
                              (2 ** bits) - 1, 0, 1, 2, 3],
                             dtype=self.utype_map[utype])).all()
            hdul.writeto(self.temp('tempfile1.fits'))

            with fits.open(self.temp('tempfile1.fits'), uint=True) as hdul1:
                assert (hdul[hdu_number].data == hdul1[hdu_number].data).all()
                if not compressed:
                    # TODO: Enable these lines if CompImageHDUs ever grow .section
                    # support
                    assert (hdul[hdu_number].section[:1].dtype.name ==
                            'uint{0:d}'.format(bits))
                    assert (hdul[hdu_number].section[:1] ==
                            hdul[hdu_number].data[:1]).all()

    @pytest.mark.parametrize(('utype',), [('u2',), ('u4',), ('u8',)])
    def test_uint_columns(self,utype):
        bits = 8 * int(utype[1])

        # Construct array
        bzero = self.utype_map[utype](2**(bits-1))
        one = self.utype_map[utype](1)
        u0 = np.arange(bits + 1, dtype=self.utype_map[utype])
        u = 2**u0 - one
        if bits == 64:
            u[63] = bzero - one
            u[64] = u[63] + u[63] + one
        uu = (u - bzero).view(self.itype_map[utype])

        # Construct a table from explicit column
        col = fits.Column(name=utype, array=u, format=self.format_map[utype],
                          bzero=bzero)
        table = fits.new_table([col])
        assert (table.data[utype] == u).all()
        assert (table.data.base[utype] == uu).all()
        hdu0 = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu0, table])
        hdulist.writeto(self.temp('tempfile.fits'))

        # Test write of unsigned int
        del hdulist
        with fits.open(self.temp('tempfile.fits'), uint=True) as hdulist2:
            hdudata = hdulist2[1].data
            assert (hdudata[utype] == u).all()
            assert (hdudata[utype].dtype == self.utype_map[utype])
            assert (hdudata.base[utype] == uu).all()

        # Test that opening the file without uint=True still returns signed
        # ints
        with fits.open(self.temp('tempfile.fits')) as hdulist2:
            hdudata = hdulist2[1].data
            assert (hdudata[utype].dtype.name == 'int{0:d}'.format(bits))

        # Construct recarray then write out that.
        v = u.view(dtype=[(utype,self.utype_map[utype])])
        fits.writeto(self.temp('tempfile2.fits'), v)
        with fits.open(self.temp('tempfile2.fits'), uint=True) as hdulist3:
            hdudata3 = hdulist3[1].data
            assert (hdudata3.base[utype] == table.data.base[utype]).all()
            assert (hdudata3[utype] == table.data[utype]).all()
            assert (hdudata3[utype] == u).all()
