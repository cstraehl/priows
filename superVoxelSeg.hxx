#pragma once
#ifndef TWS_ITER_HXX
#define TWS_ITER_HXX

#include <Python.h>
#include <boost/python.hpp>

#include <vector>
#include <queue>
#include <map>

//#include <ext/hash_map>

//#include <google/sparse_hash_map>
//#include <google/dense_hash_map>

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
#include "helpers.hxx"


//random walker
#include <list>
extern "C" {
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <lis.h>

void real_error_function(char * s) {
    printf(s);
}
}

namespace std
{
 //using namespace __gnu_cxx;
}



namespace vigra
{

namespace incremental {
     

template<class TYPE_IND, class CT>
class AdjacencyListVertex;

template<class TYPE_IND, class CT>
class AdjacencyListEdge;

template<class TYPE_IND, class CT>
class AdjacencyListEdge {
  public:
    AdjacencyListEdge() {
      min = 0;
      other = 0;
      here = 0;
      sum = 0;
      size = 0;
    }

    AdjacencyListEdge(CT cweight, float suma, unsigned int sizea, TYPE_IND cother, TYPE_IND chere) {
      other = cother;
      here = chere;
      min = cweight;
      sum = suma;
      size = sizea;
    }

    CT min;
    float sum;
    unsigned int size;
    TYPE_IND other;
    TYPE_IND here;

};

template<class TYPE_IND, class CT>
class AdjacencyListVertex {
  public:
    std::vector< AdjacencyListEdge<TYPE_IND, CT > > edges;
    unsigned int totalSize;
    //float average;
    //float price;
    //TYPE_IND volume;
    //TYPE_IND joinedWith;
};

template<class TYPE_IND, class CT>
bool edgeSortFunction(const AdjacencyListEdge<TYPE_IND, CT>& a, const AdjacencyListEdge<TYPE_IND, CT>& b) {
  CT amin =a.min;
  CT bmin=b.min;
  return amin<bmin;
}

template<class TYPE_IND, class CT>
class edgeSortClass
{
public:
  edgeSortClass()
  {}
  bool operator() (AdjacencyListEdge<TYPE_IND, CT>* a, AdjacencyListEdge<TYPE_IND, CT>* b) const
  {
    return a->min > b->min;
  }
};

template<class TYPE_IND, class PIXEL_TYPE>
struct PassInfo{
  PassInfo(TYPE_IND  b  = 0, PIXEL_TYPE weight=0) : b(b), weight(weight) {}
  TYPE_IND b;
  PIXEL_TYPE weight;

  bool operator<(const PassInfo<TYPE_IND, PIXEL_TYPE> &b) const
  {
    if (this->b < b.b) {
      return true;
    }
    else if(this->b > b.b) {
      return false;
    }
    else if (this->b == b.b){
      if(this->weight >= b.weight){
        return false;
      }
      else {
        return true;
      }
    }
  }
};


struct eqtypeind{
  bool operator()(unsigned int s1, unsigned int s2) const {
    return s1 == s2;
  }
};

struct equint{
  bool operator()(unsigned int s1, unsigned int s2) const {
    return s1 == s2;
  }
};

struct eqfloat{
  bool operator()(float s1, float s2) const {
    return s1 == s2;
  }
};

//using google;
//using __gnu_cxx::hash;

/**
* SupervoxelSegmentation class
*
* class that wraps an incremental watershed by first calculating all local
* minima and doing a normal watershed to identify all catchment basins.
*
* then a graph representing these catchment basins and thier overflow borders is
* constructed to be able to do a sublinear watershed on user provided labels.
*/
template<class TYPE_IND, class CT, class CT2, class ST>
class SupervoxelSegmentation {
  private:
    typedef typename vigra::NumpyArray<1, CT, ST >::difference_type oneDShape;
    typedef typename vigra::NumpyArray<3, TYPE_IND, ST >::difference_type threeDShape;
    //typedef typename std::hash_map< TYPE_IND, float > myMapTypeFloat;
    //typedef typename google::sparse_hash_map< unsigned int, float, hash< unsigned int>, equint> myMapTypeIndNew;
    typedef typename std::map< TYPE_IND, PassInfo<TYPE_IND, CT>*> myMapTypeInfo;
    
    vigra::NumpyArray<3, TYPE_IND, ST > _volumeBasins;
    TYPE_IND _minimaCount;
    //std::vector< myMapTypeFloat > _passHoehe;
    //std::vector< myMapTypeInd > _passSize;
    //std::vector< myMapTypeInd > _passTotalHoehe;
    std::vector< AdjacencyListVertex< TYPE_IND, CT > > _graph;
    std::vector< AdjacencyListEdge< TYPE_IND,CT > > _cachedEdges;

    vigra::NumpyArray<1, unsigned char, ST> basinSeedNumber;

    vigra::NumpyArray<1, TYPE_IND, ST > _adjustedIndices;
    
    std::vector<unsigned char>  _labelNumbers;
    
    std::vector<bool> _isAtVolumeBorder;

    std::vector< TYPE_IND > _labelBasins;
    
    std::vector< std::vector<PassInfo<TYPE_IND, CT> > > _neighbourhoods;
    
    CT _minimumHeight;
    CT  _maximumHeight;
    CT _realMinimumHeight;
    CT  _realMaximumHeight;
    int _queueCount;    
    unsigned int _numEdges;
    unsigned char maxLabel;
    bool useDifference;
    vigra::NumpyArray<3, CT, ST> _vol;
    
    vigra::NumpyArray<2, float, ST> _potentials;
    
    
    
    /**
    * private helper function that does the initial watershed and labeling of the basins
    */
    void constructBasins ( const vigra::NumpyArray<3, CT, ST>& vol, bool dontUseSuperVoxels){
        _graph.clear();
        //_vol.reshape(vol.shape());
        _vol = vol;

        // define _queueCount queues, one for each gray level.
        std::vector<std::queue<vigra::MultiArrayIndex> > queues;
        queues.resize(_queueCount+1);                
        _volumeBasins.reshape(vol.shape());
        
        if(!dontUseSuperVoxels) {
            printf("Constructing supervoxels :\n");

            //calculate the watershed seeds, and number of minima
            _minimaCount = UniqueLocalMinima3D<CT, TYPE_IND, ST >(vol, _volumeBasins);

            printf("Number of supervoxels %d   -->  compression ratio: %f\n",(int)_minimaCount, (float) _minimaCount / (vol.shape(0)*vol.shape(1)*vol.shape(2)));


                unsigned int quenr;
                //printf("Adding Minimas...\n");
                typename vigra::MultiArrayView<3, CT >::difference_type normalShape = vol.shape();
                //printf("Shape [%d, %d, %d]\n", normalShape[0],normalShape[1],normalShape[2]);

                // add each minimum
                // to the queue corresponding to its gray level
                for(vigra::MultiArrayIndex j = 0; j < _volumeBasins.size(); ++j) {
                typename vigra::NumpyArray<3, CT, ST >::difference_type coordinate = vol.scanOrderIndexToCoordinate(j);
                if(_volumeBasins[j] != 0) {
                    //printf("Adding minima [%d, %d, %d] : %d\n", coordinate[0],coordinate[1],coordinate[2], _volumeBasins[j]);
                    quenr = (float)(vol[j] - _minimumHeight)*((float)_queueCount/_maximumHeight);
                    queues[quenr].push(j);
                    _volumeBasins[j] = _volumeBasins[j] - 1;
                    //_graph[_volumeBasins[j]].color = vol[j];
                }

                }

        }
        else{
            _minimaCount = vol.shape(0)*vol.shape(1)*vol.shape(2);
            TYPE_IND count = 0;
            
            for(TYPE_IND i = 0; i< vol.shape(0); ++i){
                for(TYPE_IND j = 0; j< vol.shape(1); ++j){
                    for(TYPE_IND k = 0; k< vol.shape(2); ++k){
                        _volumeBasins[count] = count;
                        queues[0].push(count);
                        count++;
                    }   
                }
            }
        }
        
        int t1 = clock();

        unsigned int maxCount = vol.shape(0)*vol.shape(1)*vol.shape(2);
        unsigned int onePercent = maxCount / 100;
        unsigned int tenPercent = maxCount / 10;
        unsigned char percents = 0;
        unsigned int currentCount = 0;

        printf("Flooding...\n");

        _neighbourhoods.clear();
        _neighbourhoods.resize(_minimaCount);
        
        // flood
        int grayLevel = 0;
        for(;;) {
          while(!queues[grayLevel].empty()) {
            // label pixel and remove from queue
            vigra::MultiArrayIndex j = queues[grayLevel].front();
            queues[grayLevel].pop();

	    
            currentCount++;
            if(currentCount > tenPercent) {
              percents+=10;
              currentCount = 0;
              printf("finished %d\%\n",percents);
            }
	    
	    
            // add unlabeled neighbors to queues
            // left, upper, and front voxel
            typename vigra::NumpyArray<3, CT, ST >::difference_type coordinate = vol.scanOrderIndexToCoordinate(j);
            typename vigra::NumpyArray<3, CT, ST >::difference_type coordinate_temp = vol.scanOrderIndexToCoordinate(j);
            
            _realMinimumHeight = std::min(vol[coordinate_temp],_realMinimumHeight);
            _realMaximumHeight = std::max(vol[coordinate_temp], _realMaximumHeight);
            
            
            for(unsigned short d = 0; d<3; ++d) {
              if(coordinate[d] != 0) {
                --coordinate[d];
                TYPE_IND l1 = std::max(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                TYPE_IND l2 = std::min(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                CT curPixelLevel;
                if(useDifference) {
                    curPixelLevel = abs((vol[coordinate] - vol[coordinate_temp])/2);
                }
                else{
                    curPixelLevel = (vol[coordinate] + vol[coordinate_temp])/2;
                }
                int curWeight = std::max((int)((curPixelLevel - _minimumHeight)*(_queueCount/_maximumHeight)), grayLevel);
                if(_volumeBasins[coordinate] == 0) {
                  vigra::MultiArrayIndex k = vol.coordinateToScanOrderIndex(coordinate);
                  _volumeBasins[k] = _volumeBasins[j]; // label pixel
                  queues[curWeight].push(k);
                }
                //when neighbour node is already labeled, check for lowest passHoehe
                else if(l1 > l2) {
                  _neighbourhoods[l2].push_back(PassInfo<TYPE_IND,CT>(l1,curPixelLevel));
                }
                ++coordinate[d];
                }
            }
            // right, lower, and back voxel
            for(unsigned short d = 0; d<3; ++d) {
              if(coordinate[d] < _volumeBasins.shape(d)-1) {
                ++coordinate[d];
                TYPE_IND l1 = std::max(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                TYPE_IND l2 = std::min(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                CT curPixelLevel;
                if(useDifference) {
                    curPixelLevel = abs((vol[coordinate] - vol[coordinate_temp])/2);
                }
                else{
                    curPixelLevel = (vol[coordinate] + vol[coordinate_temp])/2;
                }                int curWeight = std::max((int)((curPixelLevel - _minimumHeight)*(_queueCount/_maximumHeight)), grayLevel);
                if(_volumeBasins[coordinate] == 0) {
                  vigra::MultiArrayIndex k = vol.coordinateToScanOrderIndex(coordinate);
                  _volumeBasins[k] = _volumeBasins[j]; // label pixel
                  queues[curWeight].push(k);
                }
                //when neighbour node is already labeled, check for lowest passHoehe
                else if(l1 > l2) {
                  _neighbourhoods[l2].push_back(PassInfo<TYPE_IND,CT>(l1,curPixelLevel));
                }
              --coordinate[d];
              }
            }
          }
          if(grayLevel == _queueCount) {
            break;
          }
          else {
            //queues[grayLevel].clear(); // free memory
            ++grayLevel;
          }
        }
        queues.clear();
        printf("MinimumLocalWeight: %d, MaximumLocalWeight: %d\n",_realMinimumHeight,_realMaximumHeight);
        printf("done.\n");
        
        #pragma omp parallel for
        for(int i = 0; i<_neighbourhoods.size();++i){
          sort(_neighbourhoods[i].begin(),_neighbourhoods[i].end());
        }

      //detect which basins lie at the volume border
      _isAtVolumeBorder.resize(0);
      _isAtVolumeBorder.resize(_minimaCount, false);

      typedef NeighborCode3DSix Neighborhood;
      typedef Neighborhood::Direction Direction;
      MultiArrayShape<3>::type p(0,0,0);


      for(p[2]=0; p[2]<vol.shape(2); p[2]+= vol.shape(2)-1)
      {
        for(p[1]=0; p[1]<vol.shape(1); ++p[1])
        {
          for(p[0]=0; p[0]<vol.shape(0); ++p[0])
          {
          TYPE_IND basin = _volumeBasins[p];
          _isAtVolumeBorder[basin] = true;
          }
        }
      }

      for(p[2]=0; p[2]<vol.shape(2); ++p[2])
      {
        for(p[1]=0; p[1]<vol.shape(1); p[1]+= vol.shape(1)-1)
        {
          for(p[0]=0; p[0]<vol.shape(0); ++p[0])
          {
          TYPE_IND basin = _volumeBasins[p];
          _isAtVolumeBorder[basin] = true;
          }
        }
      }

      for(p[2]=0; p[2]<vol.shape(2);  ++p[2])
      {
        for(p[1]=0; p[1]<vol.shape(1); ++p[1])
        {
          for(p[0]=0; p[0]<vol.shape(0); p[0]+= vol.shape(0)-1)
          {
          TYPE_IND basin = _volumeBasins[p];
          _isAtVolumeBorder[basin] = true;
          }
        }
      }      


      //build the watershed graph
      buildWSGraph();

      int t2 = clock();

      double time = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;

      printf("Preprocessing() took: %f\n", time);
        
    }


      public:
          
          
  void setSeeds(const vigra::MultiArrayView<1, vigra::UInt8, ST>& labelNumbers, const vigra::MultiArrayView<1, TYPE_IND, ST>& labelIndices) {


      /*
      * begin really stupid c/fortran order indices conversion
      */
      _labelNumbers.resize(labelNumbers.shape(0));
      for(int i=0;i<_labelNumbers.size();++i) {
          _labelNumbers[i] = labelNumbers[i];
      }
      
      _adjustedIndices.reshape(labelIndices.shape());

      typename vigra::MultiArrayView<3, TYPE_IND, ST >::difference_type switchedShape = _volumeBasins.shape();
      //printf("Original Shape [%d, %d, %d]\n", switchedShape[0],switchedShape[1],switchedShape[2]);
      switchedShape[0] = switchedShape[2];
      switchedShape[2] = _volumeBasins.shape()[0];

      for(int i = 0; i < _adjustedIndices.size(); ++i) {
        vigra::MultiArrayView<3, vigra::UInt32>::difference_type coordinate;
        vigra::detail::ScanOrderToCoordinate<3>::exec(labelIndices(i), switchedShape, coordinate);

        unsigned int tempI = coordinate[0];
        coordinate[0] = coordinate[2];
        coordinate[2] = tempI;
        //printf("CoordinateA [%d, %d, %d] : %d, basin %d\n", coordinate[0],coordinate[1],coordinate[2],labelNumbers[i], _volumeBasins[coordinate]);
        _adjustedIndices(i) = _volumeBasins.coordinateToScanOrderIndex(coordinate);
      }
      /*
      * end really stupid c/fortran order indices conversion
      */

      std::vector<bool> labelCollision;
      labelCollision.resize(_minimaCount, false);
      
      bool labelCollisions = true;
      
      while(labelCollisions) {
        labelCollisions = false;
        
        //reset basinSeedNumbers
        basinSeedNumber.reshape(oneDShape(_minimaCount)); //make room for 2 labels : border and bridges
        for(TYPE_IND i=0; i < _minimaCount; ++i) {
            basinSeedNumber(i) = 0;
        }
        
        //reset maxLabel
        maxLabel = 0;
        
        for(TYPE_IND i = 0; i < _labelNumbers.size(); ++i) {
            TYPE_IND index = _adjustedIndices[i];
            unsigned char label = _labelNumbers[i];
            vigra::MultiArrayView<3, vigra::UInt32>::difference_type coordinate = _volumeBasins.scanOrderIndexToCoordinate(index);

            TYPE_IND basin = _volumeBasins[coordinate];
            maxLabel = std::max(label,maxLabel);

            //check for seed collision in basins
            if ( (basinSeedNumber[basin] == 0 || basinSeedNumber[basin] == label)) {
                basinSeedNumber[basin] = label;
            }
            else {
                labelCollisions = true;
                handleLabelCollision(basin, coordinate);
                printf("recalculating seeds\n");
                break;
            }
        }
      }



      // determine the basins for each labeled pixel
       _labelBasins.resize(_minimaCount);

      for(TYPE_IND i = 0; i < _labelNumbers.size(); ++i) {
        TYPE_IND index = _adjustedIndices[i];
        unsigned char label = _labelNumbers[i];
        vigra::MultiArrayView<3, vigra::UInt32>::difference_type coordinate = _volumeBasins.scanOrderIndexToCoordinate(index);

        _labelBasins[i] = _volumeBasins[coordinate];
      }


  }


  void addVirtualBackgroundSeeds(float bias, float biasThres, unsigned int biasedLabel) {
      vigra::NumpyArray<1, unsigned char, ST> tempBasinSeedNumber;

      biasThres = biasThres * bias;
      bias = bias*bias;
      
      //reset tempBasinSeedNumbers
      tempBasinSeedNumber.reshape(oneDShape(_minimaCount)); //make room for 2 labels : border and bridges
      for(TYPE_IND i=0; i < _minimaCount; ++i) {
        tempBasinSeedNumber(i) = 0;
      }


      //reset tempBasinSeedNumbers
      tempBasinSeedNumber.reshape(oneDShape(_minimaCount)); //make room for 2 labels : border and bridges
      for(TYPE_IND i=0; i < _minimaCount; ++i) {
        tempBasinSeedNumber(i) = 0;
      }

      // define a priority queue
      std::priority_queue< AdjacencyListEdge<TYPE_IND, CT>*, std::vector<AdjacencyListEdge<TYPE_IND, CT >* >, edgeSortClass<TYPE_IND, CT> > queue;

      unsigned char maxLabel = 0;

      std::vector< AdjacencyListVertex< TYPE_IND, CT > > graphCopy = _graph;

      // define 2048 queues, one for each value.
      std::vector<std::queue<AdjacencyListEdge<TYPE_IND, CT>*> > queues;
      queues.resize(_queueCount+1);


      for(TYPE_IND i = 0; i < _labelNumbers.size(); ++i) {
        unsigned char label = _labelNumbers[i];

        TYPE_IND basin = _labelBasins[i];
        maxLabel = std::max(label,maxLabel);

        tempBasinSeedNumber[basin] = label;
        // add each unlabeled basin bordering a labeled basin to the priority queue
        AdjacencyListVertex< TYPE_IND, CT >* v = &graphCopy[basin];
        for(TYPE_IND j=0; j < v->edges.size();++j) {
        AdjacencyListEdge< TYPE_IND, CT > *e = &v->edges[j];

        if(tempBasinSeedNumber[e->other] == 0) {
            //printf("weight %d\n", e->weight);
            int weight = e->min;
            queues[weight].push(e);
        }
        }
      }

      int waterLevel = 0;


      for(;;){
        while(!queues[waterLevel].empty()) {
          // label pixel and remove from queue
          AdjacencyListEdge<TYPE_IND, CT>* e1 = queues[waterLevel].front();
          queues[waterLevel].pop();
          vigra::MultiArrayIndex i = e1->other;
          float factor = 1.0;
          TYPE_IND minVertex;
          AdjacencyListEdge< TYPE_IND, CT >* e;



          if(tempBasinSeedNumber[i] == 0) { //still unlabeled ?
            tempBasinSeedNumber[i] = tempBasinSeedNumber[e1->here];

            if(tempBasinSeedNumber[i] == 0) {
                printf("hae?: %d\n", i);
            }

            if(tempBasinSeedNumber[i] == biasedLabel) {
              factor = bias;
            }

            //graphCopy[i].price = std::max(graphCopy[e1->here].price,e1->weight * factor);

            AdjacencyListVertex< TYPE_IND, CT >* v = &(graphCopy[i]);

            for(TYPE_IND j=0; j < v->edges.size();++j) {
                e = &(v->edges[j]);
                if(tempBasinSeedNumber[e->other] == 0) { //unlabeled
                  int weight;
                  if(e->min > biasThres) {
                    weight = std::max(waterLevel, (int)(e->min * factor));
                    //printf("weight: %d\n", weight);
                  }
                  else {
                    weight = std::max(waterLevel, (int)(e->min));
                  }
                queues[weight].push(e);
                }
              }
          }
        }
        if(waterLevel==_queueCount){
          break;
        }
        else{
          waterLevel++;
        }
      }




      TYPE_IND __count = 0;
      for(TYPE_IND i=0; i < _minimaCount; ++i) {
        if(_isAtVolumeBorder[i] and tempBasinSeedNumber[i] == biasedLabel) {
        _labelNumbers.push_back(biasedLabel);
        _labelBasins.push_back(i);
        basinSeedNumber[i] = biasedLabel;
        __count++;
        }
      }
      printf("Seeded an additional %d number of volume border basins\n", __count);
  }

          
 void handleLabelCollision(TYPE_IND basin, typename vigra::MultiArrayView<3, vigra::UInt32>::difference_type coordinate) {
    typedef typename vigra::NumpyArray<3, CT, ST >::difference_type VCOORD;

    printf("Seed Collision in Basin %d, splitting up basin\n", basin);
    
    VCOORD coordinate_temp = coordinate;
    std::queue<VCOORD> basinPixels;
    TYPE_IND basinPixelCount = 0;
    basinPixels.push(coordinate);
    TYPE_IND maxBasin = std::numeric_limits<TYPE_IND>::max();
    
    _volumeBasins[coordinate] = maxBasin;
    
    printf("a\n");
    
    while(!basinPixels.empty()){
        coordinate_temp = basinPixels.front();
        
        basinPixels.pop();
        
        basinPixelCount++;
        for(unsigned short d = 0; d<3; ++d) {
            if(coordinate_temp[d] != 0) {
                --coordinate_temp[d];
                if(_volumeBasins[coordinate_temp] == basin) {
                   _volumeBasins[coordinate_temp] = maxBasin;
                   basinPixels.push(coordinate_temp);
                }
                ++coordinate_temp[d];
            }
        }
        for(unsigned short d = 0; d<3; ++d) {
            if(coordinate_temp[d] < _volumeBasins.shape(d)-1) {
                ++coordinate_temp[d];
                if(_volumeBasins[coordinate_temp] == basin) {
                   _volumeBasins[coordinate_temp] = maxBasin; 
                   basinPixels.push(coordinate_temp);
                }                
                --coordinate_temp[d];
            }
        }
    }
    printf("b\n");
    
    TYPE_IND oldMinimaCount = _minimaCount;
    
    //adjust minimaCount
    _minimaCount += basinPixelCount;
    printf("c\n");
            
    //remove obsolete entries from the neighborhood lists
    std::vector< std::vector<PassInfo<TYPE_IND, CT> > > tempNeighbourhoods;
    tempNeighbourhoods.resize(_minimaCount); //NEW minimum count
    //printf("d\n");
    
    #pragma omp parallel for
    for(int i = 0; i<oldMinimaCount;++i){
        tempNeighbourhoods[i] = _neighbourhoods[i];
    }    
    printf("e\n");
    
    //delete and resize old neighborhoods
    _neighbourhoods.clear();
    _neighbourhoods.resize(_minimaCount); //new minima count

    printf("f\n");
    
    //renumber the basin pixels and add new passinfos
    TYPE_IND currentMinimum = oldMinimaCount;    
    _volumeBasins[coordinate] = currentMinimum;
    currentMinimum++;

    basinPixels.push(coordinate);
    printf("g\n");
    
    while(!basinPixels.empty()){
        
        coordinate_temp = basinPixels.front();
        coordinate = coordinate_temp;
        
        basinPixels.pop();
                
        for(unsigned short d = 0; d<3; ++d) {
            if(coordinate_temp[d] != 0) {
                --coordinate_temp[d];
                if(_volumeBasins[coordinate_temp] == maxBasin) {
                    //also relabel so that it does not get added again
                   _volumeBasins[coordinate_temp] = currentMinimum;
                   currentMinimum++;
                   basinPixels.push(coordinate_temp);
                }
                TYPE_IND l1 = std::max(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                TYPE_IND l2 = std::min(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                
                CT curPixelLevel;
                
                //printf("naja\n");
                if(useDifference) {
                    curPixelLevel = abs((_vol[coordinate] - _vol[coordinate_temp])/2);
                }
                else{
                    curPixelLevel = (_vol[coordinate] + _vol[coordinate_temp])/2;
                }                
                //printf("naja2\n");
                
                if(l1 > l2) {
                  _neighbourhoods[l2].push_back(PassInfo<TYPE_IND,CT>(l1,curPixelLevel));
                }
                
                ++coordinate_temp[d];
            }
        }
        for(unsigned short d = 0; d<3; ++d) {
            if(coordinate_temp[d] < _volumeBasins.shape(d)-1) {
                ++coordinate_temp[d];
                if(_volumeBasins[coordinate_temp] == maxBasin) {
                    //also relabel so that it does not get added again
                   _volumeBasins[coordinate_temp] = currentMinimum; 
                   currentMinimum++;
                   basinPixels.push(coordinate_temp);
                }                

                TYPE_IND l1 = std::max(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                TYPE_IND l2 = std::min(_volumeBasins[coordinate],_volumeBasins[coordinate_temp]) ;
                
                CT curPixelLevel;
                //printf("naja\n");
                
                if(useDifference) {
                    curPixelLevel = abs((_vol[coordinate] - _vol[coordinate_temp])/2);
                }
                else{
                    curPixelLevel = (_vol[coordinate] + _vol[coordinate_temp])/2;
                }                
                //printf("naja2\n");
                
                if(l1 > l2) {
                  _neighbourhoods[l2].push_back(PassInfo<TYPE_IND,CT>(l1,curPixelLevel));
                }

                --coordinate_temp[d];
            }
        }
    }    
    
    
    printf("h\n");
    
    //copy over old + new passinfos, while removing obsolete connections
    #pragma omp parallel for
    for(int i = 0; i<_neighbourhoods.size();++i){ //OLD minimum count
        if(i!=basin) {
            PassInfo<TYPE_IND, CT> passinfo;
            for(int j=0; j<tempNeighbourhoods[i].size();++j){
                passinfo = tempNeighbourhoods[i][j];
                if(passinfo.b != basin) {
                    _neighbourhoods[i].push_back(passinfo);
                }
            }
        }
    }
    
    printf("i\n");
    
    //sort stuff again
    #pragma omp parallel for
    for(int i = 0; i<_neighbourhoods.size();++i){
        sort(_neighbourhoods[i].begin(),_neighbourhoods[i].end());
    }    

    //rebuild graph
    buildWSGraph();
 }



  void buildWSGraph() {
          printf("Constructing graph for Watershed ");

          printf("...\n");

          //allocate adjacency graph
          _graph.clear();
          _graph.resize(_neighbourhoods.size());

          _numEdges = 0;
          TYPE_IND lastB=0;
          TYPE_IND lastI=0;
          CT weight;
          float sum = 0;

          unsigned int size = 0;
          unsigned int totalSize = 0;
          CT minimumW ;

          //construct the basin graph
          for(int i = 0; i<_neighbourhoods.size();++i){
              typename std::vector<PassInfo<TYPE_IND, CT> >::iterator it;
              typename std::vector<PassInfo<TYPE_IND, CT> >::iterator end = _neighbourhoods[i].end();

              totalSize = 0;

              if(_neighbourhoods[i].size() > 0) {
                  lastB = _neighbourhoods[i].begin()->b;
                  minimumW = _neighbourhoods[i].begin()->weight;
                  sum = 0;
                  size = 0;
              }

              for(it = _neighbourhoods[i].begin(); it  != end; ++it) {
                  if(lastB != it->b){
                      //printf("Edge %d ->, %d: %f\n", i, it->first, weight);

                      //(CT)((it->weight - _minimumHeight) * (float) _queueCount / _maximumHeight);
                      AdjacencyListEdge< TYPE_IND,CT > e1(minimumW, sum, size, i, lastB);
                      AdjacencyListEdge< TYPE_IND,CT > e2(minimumW, sum, size, lastB, i);
                      _graph[lastB].edges.push_back(e1);
                      _graph[i].edges.push_back(e2);
                      _numEdges +=2;
                      lastB = it->b;
                      lastI = i;
                      minimumW = it->weight;

                      sum = 0;
                      size = 0;
                  }
                  sum += it->weight;
                  size++;
                  totalSize++;
              }

              if(_neighbourhoods[i].size() > 0) {
                  AdjacencyListEdge< TYPE_IND,CT > e1(minimumW, sum, size, i, lastB);
                  AdjacencyListEdge< TYPE_IND,CT > e2(minimumW, sum, size, lastB, i);
                  _graph[lastB].edges.push_back(e1);
                  _graph[i].edges.push_back(e2);
              }
              _graph[i].totalSize = totalSize;
              lastI = i;

          }
      printf(" done !\n");

  }

  /**
  * public interface method that does the watershed on the sparse graph.
  *
  * Input: two 1D array consisting of the label or seed Numbers, and their corresponding 3d scanOrderIndices
  *
  * Output: nothing
  *
  * Returns: 	array containg the calculated labelNumbers of the various catchment basins
  * 		should be used as an index table for the basin
  *
  */
  NumpyAnyArray doWS(float bias, float biasThres, unsigned int biasedLabel, bool virtualBackgroundSeeds) {


      if(virtualBackgroundSeeds) {
        addVirtualBackgroundSeeds(bias, biasThres, biasedLabel);
      }

      printf("Executing watershed on %d Vertices and %d Edges...\n", (int) _minimaCount, (int) _numEdges);
      
      
      unsigned int t1 = clock();

      //printf("doing supervoxel watershed with bias %f for label %d\n",_bias, _biasedLabel);
      
      std::vector< AdjacencyListVertex< TYPE_IND, CT > > graphCopy = _graph;

      //reset basinSeedNumbers
      basinSeedNumber.reshape(oneDShape(_minimaCount)); //make room for 2 labels : border and bridges
      for(TYPE_IND i=0; i < _minimaCount; ++i) {
        basinSeedNumber(i) = 0;
      }
      
      // define a priority queue
      std::priority_queue< AdjacencyListEdge<TYPE_IND, CT>*, std::vector<AdjacencyListEdge<TYPE_IND, CT >* >, edgeSortClass<TYPE_IND, CT> > queue;

      unsigned char maxLabel = 0;


      // define 2048 queues, one for each value.
      std::vector<std::queue<AdjacencyListEdge<TYPE_IND, CT>*> > queues;
      queues.resize(_queueCount+1);

      
      for(TYPE_IND i = 0; i < _labelNumbers.size(); ++i) {
        unsigned char label = _labelNumbers[i];
        TYPE_IND basin = _labelBasins[i];
        maxLabel = std::max(label,maxLabel);

        basinSeedNumber[basin] = label;
        // add each unlabeled basin bordering a labeled basin to the priority queue
        AdjacencyListVertex< TYPE_IND, CT >* v = &graphCopy[basin];
        for(TYPE_IND j=0; j < v->edges.size();++j) {
        AdjacencyListEdge< TYPE_IND, CT > *e = &v->edges[j];

        if(basinSeedNumber[e->other] == 0) {
            //printf("weight %d\n", e->weight);
            int weight = e->min;
            queues[weight].push(e);
        }
        }
      }

      int waterLevel = 0;


      for(;;){
        while(!queues[waterLevel].empty()) {
          // label pixel and remove from queue
          AdjacencyListEdge<TYPE_IND, CT>* e1 = queues[waterLevel].front();
          queues[waterLevel].pop();
          vigra::MultiArrayIndex i = e1->other;
          float factor = 1.0;
          TYPE_IND minVertex;
          AdjacencyListEdge< TYPE_IND, CT >* e;

          
          
          if(basinSeedNumber[i] == 0) { //still unlabeled ?
            basinSeedNumber[i] = basinSeedNumber[e1->here];

            if(basinSeedNumber[i] == 0) {
                printf("hae?: %d\n", i);
            }
            
            if(basinSeedNumber[i] == biasedLabel) {
              factor = bias;
            }

            //graphCopy[i].price = std::max(graphCopy[e1->here].price,e1->weight * factor);

            AdjacencyListVertex< TYPE_IND, CT >* v = &(graphCopy[i]);

            for(TYPE_IND j=0; j < v->edges.size();++j) {
                e = &(v->edges[j]);
                if(basinSeedNumber[e->other] == 0) { //unlabeled
                  int weight;
                  if(e->min > biasThres) {
                    weight = std::max(waterLevel, (int)(e->min * factor));
                    //printf("weight: %d\n", weight);
                  }
                  else {
                    weight = std::max(waterLevel, (int)(e->min));
                  }
                queues[weight].push(e);
                }
              }
          }
        }
        if(waterLevel==_queueCount){
          break;
        }
        else{
          waterLevel++;
        }
      }

      int t2 = clock();

      double time = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;

      printf("Algorithm() took: %f\n", time);

      return basinSeedNumber;
    }




    /**
    * Public interface function - returns the calculated basinNumbers of the Volume
    *
    */
    NumpyAnyArray getVolumeBasins(vigra::NumpyArray<3, TYPE_IND, ST> res) {

      //res.reshape(_volumeBasins.shape());

      vigra::NumpyArray<3, TYPE_IND, ST> result = _volumeBasins;

      return result;
    }

    /**
    * Public interface function - returns the potentials of the basins regarding the labels
    *
    */
    NumpyAnyArray getBasinPotentials() {
      vigra::NumpyArray<2, float, ST> res = _potentials;
      
      return res;
    }

    /**
    * Public interface function - returns the calculated basinNumbers of the Volume
    *
    */
    NumpyAnyArray getBorderVolume() {

      vigra::NumpyArray<3, unsigned char, ST> res;
      res.reshape(_volumeBasins.shape());

      //mark borders
      for(int j=0; j < _volumeBasins.size();++j) {
        typename vigra::NumpyArray<3, TYPE_IND, ST >::difference_type coordinate = _volumeBasins.scanOrderIndexToCoordinate(j);
        for(unsigned short d = 0; d<3; ++d) {
          if(coordinate[d] != 0) {
            --coordinate[d];
            if(_volumeBasins[coordinate] != _volumeBasins[j]) {
              res[j] = 1;
            }
            else {
                res[j] = 0;
            }
          ++coordinate[d];
          }
        } 
      }
      return res;
    }


    /**
    * constructor, calculates the local minimy and catchment basins of the provided volume
    *
    * Input: 	a 3d Volume of unsigned chars
    *
    * Output:	nothing
    */
    SupervoxelSegmentation(const vigra::NumpyArray<3, CT, ST>& vol, bool useDiff, float minimum, float maximum, int queuecount, bool dontUseSuperVoxels)
    {
      //construct the basin graph, and label the volume according to the watershed algorithm from the minima
      _minimumHeight = minimum;
      _maximumHeight = maximum;
      _realMinimumHeight = std::numeric_limits<CT>::max();
      _realMaximumHeight = 0;
      _queueCount = queuecount;
      useDifference = useDiff;
      constructBasins(vol, dontUseSuperVoxels);
    }


    /**
    * destructor,
    */
    ~SupervoxelSegmentation() {
      //clear stuff
/*	    for(TYPE_IND i = 0; i < _cachedEdges.size();++i) {
        delete _cachedEdges[i];
      }*/
      _cachedEdges.clear();

/*	    for(TYPE_IND i = 0; i < graphCopy.size();++i) {
        AdjacencyListVertex< TYPE_IND >* v = &_graph[i];

        for(TYPE_IND j=0; j < v->edges.size();++j) {
    delete v->edges[j];
        }
      }*/
      _graph.clear();

    }
};

} //end namespace

}//namespce vigra

#endif
