#pragma once
#ifndef TWS_HELPER_HXX
#define TWS_HELPER_HXX

#include <Python.h>
#include <boost/python.hpp>

#include <vector>
#include <queue>
#include <map>
#include <queue>

#include <algorithm> // max
#include <map>

#include <vigra/windows.h>
#include <time.h>

#include <vigra/multi_array.hxx>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/multi_pointoperators.hxx>

#include "tws.hxx"



namespace vigra
{


  /**
  * Input: image : 3d Volume
  *     res : empty result volume of equal size
  *
  * Output: res : unique markers for local minimas
  *
  * Return: number of local minima found
  *
  *
  */
  template <class PixelType, class TYPE_IND, class ST >
  TYPE_IND UniqueLocalMinima3D(const NumpyArray<3, PixelType, ST >& image,
                              NumpyArray<3, TYPE_IND, ST >& res)
  {

    TYPE_IND maxRegionLabel = 0;

    typedef NeighborCode3DSix Neighborhood;
    typedef Neighborhood::Direction Direction;

    MultiArrayShape<3>::type p(0,0,0);

    NumpyArray<3, TYPE_IND, ST >* labels = new  NumpyArray<3, TYPE_IND, ST >(image.shape());

    int number_of_regions =
    labelVolume(srcMultiArrayRange(image),
                destMultiArray(*labels), NeighborCode3DSix());




    // assume that a region is a extremum until the opposite is proved
    std::vector<TYPE_IND> isExtremum(number_of_regions+1, (TYPE_IND)1);

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
              if( image[p]   > image[p+Neighborhood::diff((Direction)k)] )
                isExtremum[(*labels)[p]] = 0;

            }
          }
          else
          {
/*//        mark all regions that touch the image border as non-extremum 
 
           isExtremum[labels[p]] = 0;  */
              for(int k=0; k<totalCount; ++k)
              {
                  if(image[p] > image[p+Neighborhood::diff(
                                          Neighborhood::nearBorderDirections(atBorder, k))])
                      isExtremum[(*labels)[p]] = 0;
              }
          }
        }
      }
    }


    //compress label numbers and calculate number of extremas:
    TYPE_IND extrema = 0;
    for(TYPE_IND jj = 0; jj <= number_of_regions; ++jj) {
      if(isExtremum[jj]) {
      ++extrema;
      isExtremum[jj] = extrema;
      }
    }



    //fill the output array
    for(p[2]=0; p[2]<image.shape(2); ++p[2])
    {
      for(p[1]=0; p[1]<image.shape(1); ++p[1])
      {
        for(p[0]=0; p[0]<image.shape(0); ++p[0])
        {
          if(isExtremum[(*labels)[p]])
          {
            res[p] = isExtremum[(*labels)[p]];
          }
          else {
            res[p] = 0;
          }
        }
      }
    }
    
    delete labels;
    
    isExtremum.clear();
    return extrema;
  }


template<class T>
inline bool isAtSeedBorder
(
  const vigra::MultiArrayView<3, T>& labeling,
  const vigra::MultiArrayIndex& index
)
{
  if(labeling[index] == 0) {
    return false; // not a seed voxel
  }
  else {
    typename vigra::MultiArrayView<3, vigra::UInt32>::difference_type coordinate
      = labeling.scanOrderIndexToCoordinate(index);
    // check left, upper, and front voxel for zero label
    for(unsigned short d = 0; d<3; ++d) {
      if(coordinate[d] != 0) {
        --coordinate[d];
        if(labeling[coordinate] == 0) {
          return true;
        }
        ++coordinate[d];
      }
    }
    // check right, lower, and back voxel for zero label
    for(unsigned short d = 0; d<3; ++d) {
      if(coordinate[d] < labeling.shape(d)-1) {
        ++coordinate[d];
        if(labeling[coordinate] == 0) {
          return true;
        }
        --coordinate[d];
      }
    }
    return false;
  }
}

}//namespce vigra

#endif //TWS_HXX
