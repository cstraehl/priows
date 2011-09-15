#define PY_ARRAY_UNIQUE_SYMBOL vigranumpypgm_PyArray_API
//#define NO_IMPORT_ARRAY

#include <Python.h>
#include <iostream>
#include <boost/python.hpp>
#include <set>

#include <vigra/mathutil.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_pointoperators.hxx>
#include <vigra/matrix.hxx>
#include <vigra/eigensystem.hxx>
#include <vigra/array_vector.hxx>
#include <vigra/static_assert.hxx>

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/functorexpression.hxx>

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/localminmax.hxx>
#include <vigra/labelimage.hxx>
#include <vigra/watersheds.hxx>
#include <vigra/seededregiongrowing.hxx>
#include <vigra/labelvolume.hxx>
#include <vigra/watersheds3d.hxx>
#include <vigra/seededregiongrowing3d.hxx>

#include <string>
#include <cmath>
#include <ctype.h>

#include <cmath>

#include "superVoxelSeg.hxx"

namespace python = boost::python;

namespace vigra
{

template < class PixelType >
NumpyAnyArray 
pythonExtendedLocalMinima3D(NumpyArray<3, Singleband<PixelType> > image,
                            PixelType marker = NumericTraits<PixelType>::one(),
                            NumpyArray<3, Singleband<PixelType> > res = python::object())
{
    res.reshapeIfEmpty(image.shape(), "extendedLocalMinima(): Output array has wrong shape.");

            unsigned int maxRegionLabel = 0;
            
            // determine seeds
            // FIXME: implement localMinima() for volumes
            typedef NeighborCode3DSix Neighborhood;
            typedef Neighborhood::Direction Direction;
            
            MultiArrayShape<3>::type p(0,0,0);

	    NumpyArray<3, Singleband<npy_uint32> > labels(image.shape());

	    int number_of_regions =
		labelVolume(srcMultiArrayRange(image),
		    destMultiArray(labels), NeighborCode3DSix());

		    
	    // assume that a region is a extremum until the opposite is proved
	    std::vector<unsigned char> isExtremum(number_of_regions+1, (unsigned char)1);
	      
            for(p[2]=0; p[2]<image.shape(2); ++p[2])
            {
                for(p[1]=0; p[1]<image.shape(1); ++p[1])
                {
                    for(p[0]=0; p[0]<image.shape(0); ++p[0])
                    {
                        AtVolumeBorder atBorder = isAtVolumeBorder(p, image.shape());
                        int totalCount = Neighborhood::nearBorderDirectionCount(atBorder),
                            minimumCount = 0;
                        if(atBorder == NotAtBorder)
                        {
                            for(int k=0; k<totalCount; ++k)
                            {
                                if(image[p] < image[p+Neighborhood::diff((Direction)k)])
                                   isExtremum[labels[p]] = 0;
                            }
                        }
                        else
                        {
/*// 			  mark all regions that touch the image border as non-extremum
			  isExtremum[labels[p]] = 0;	*/
                            for(int k=0; k<totalCount; ++k)
                            {
                                if(image[p] < image[p+Neighborhood::diff(
                                                        Neighborhood::nearBorderDirections(atBorder, k))])
                                    isExtremum[labels[p]] = 0;
                            }
                        }	
                    }
                }
            }
            for(p[2]=0; p[2]<image.shape(2); ++p[2])
            {
                for(p[1]=0; p[1]<image.shape(1); ++p[1])
                {
                    for(p[0]=0; p[0]<image.shape(0); ++p[0])
                    {
		      if(isExtremum[labels[p]]) 
		      {
			res[p] = marker;
		      }
		    }
		}
	    }


    return res;
}


NumpyAnyArray 
pythonTWS3D(NumpyArray<3, Singleband<vigra::UInt8> > image, NumpyArray<3, Singleband<npy_uint32> > seeds,  uint label, float prio,
                   NumpyArray<3, Singleband<npy_uint32> > res) // )python::object()
{
  res.reshapeIfEmpty(image.shape());
  //NumpyArray<3, Singleband<vigra::UInt8> > minima(image.shape());
  //pythonExtendedLocalMinima3D<vigra::UInt8>(image, NumericTraits<vigra::UInt8>::one(),minima);
  //unsigned int maxRegionLabel = labelVolumeWithBackground(srcMultiArrayRange(minima), destMultiArray(res), NeighborCode3DSix(), 0);	
  res = seeds;
  tws<npy_uint32 , vigra::StridedArrayTag>( image, res, label, prio);
  return res;
}

template<class TYPE_IND, class CT, class CT2>
incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>* pythonConstructSupervoxelSegmentation(const NumpyArray<3, CT> & vol, bool useDifference, float min, float max, TYPE_IND queuecount, bool dus) {
  incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>* ws = new incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>(vol, useDifference, min, max, queuecount, dus);
  return ws;
}

// template<class TYPE_IND>
// incremental::SupervoxelSegmentation2<TYPE_IND, vigra::StridedArrayTag>* pythonConstructSupervoxelSegmentation2(const NumpyArray<3, Singleband<vigra::UInt8> > & vol) {
//   incremental::IncrementalWatershed2<TYPE_IND,vigra::StridedArrayTag>* ws = new incremental::IncrementalWatershed2<TYPE_IND, vigra::StridedArrayTag>(vol);
//   return ws;
// }

void defineTWS()
{
    using namespace python;
    def("tws", registerConverters(&pythonTWS3D),
    		(arg("image"), arg("seeds") = python::object(), arg("label"), arg("prio"), arg("out") = python::object()),
    		"Turbowatershed\n");
    
    class_<incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32, vigra::StridedArrayTag > > incWSC("segmentor",python::no_init);
    incWSC.def("__init__", python::make_constructor(registerConverters(&pythonConstructSupervoxelSegmentation< unsigned int,vigra::UInt8, vigra::UInt32 >),
						   boost::python::default_call_policies(),
						   ( arg("volume" ), arg("useDifference"), arg("min"), arg("max"), arg("queuecount"), arg("dontUseSuperVoxels") )
				  ),
				 "Constructor::\n\n"
				 "volume : a volume of type unsinged char"
	     ).def("setSeeds", registerConverters(&incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32,vigra::StridedArrayTag >::setSeeds),
	    (arg("labelNumbers"), arg("labelIndices"))
        ).def("doWS", registerConverters(&incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32,vigra::StridedArrayTag >::doWS),
        (arg("bias"), arg("biasThreshold"), arg("biasedLabel"), arg("virtualBackgroundSeeds"))
	    ).def("getVolumeBasins", registerConverters(&incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32,vigra::StridedArrayTag >::getVolumeBasins),
	    (arg("out")=object())
        ).def("getBasinPotentials", registerConverters(&incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32,vigra::StridedArrayTag >::getBasinPotentials)
	    ).def("getBorderVolume", registerConverters(&incremental::SupervoxelSegmentation< unsigned int, vigra::UInt8, vigra::UInt32,vigra::StridedArrayTag >::getBorderVolume)
	  );
}

} // namespace vigra

using namespace vigra;
using namespace boost::python;

BOOST_PYTHON_MODULE_INIT(priows)
{
    import_vigranumpy();
    defineTWS();
}
