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

template<class TYPE_IND, class CT, class CT2>
incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>* pythonConstructSupervoxelSegmentation(const NumpyArray<3, CT> & vol, bool useDifference, float min, float max, TYPE_IND queuecount, bool dus) {
  incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>* ws = new incremental::SupervoxelSegmentation<TYPE_IND, CT, CT2,vigra::StridedArrayTag>(vol, useDifference, min, max, queuecount, dus);
  return ws;
}


void definePriows()
{
    using namespace python;
    
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
    definePriows();
}
