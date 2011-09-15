#pragma once
#ifndef TWS_HXX
#define TWS_HXX

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include <vigra/windows.h>
#include <vigra/multi_array.hxx>
#include <time.h>

template<class T, class ST>
void tws
(
	const vigra::MultiArrayView<3, vigra::UInt8, ST>&,
	vigra::MultiArrayView<3, T, ST>&
);

template<class T, class ST>
void twsc
(
	const vigra::MultiArrayView<3, vigra::UInt8, ST>&,
	vigra::MultiArrayView<3, T, ST>&,
	vigra::MultiArrayView<3, vigra::UInt8, ST>&
);

template<class T ,class ST>
inline bool isAtSeedBorder
(
	const vigra::MultiArrayView<3, T, ST>& labeling,
	const vigra::MultiArrayIndex& index
);

template<class T, class ST>
void tws
(
	const vigra::MultiArrayView<3, vigra::UInt8, ST>& vol,
	vigra::MultiArrayView<3, T, ST>& labeling,
  uint prioSeed, float factor
)
{
      int t1 = clock();


	// define 256 queues, one for each gray level.
	std::vector<std::queue<vigra::MultiArrayIndex> > queues(2048);
  float priofact = factor * 8.0;
  std::vector<float> factors(256);
  for(int i = 0; i< 256;i++) {
    factors[i] = 8.0;
  }
  factors[prioSeed] = priofact;
  
	// add each unlabeled pixels which is adjacent to a seed
	// to the queue corresponding to its gray level
	for(vigra::MultiArrayIndex j = 0; j < labeling.size(); ++j) {
		if(isAtSeedBorder<T>(labeling, j)) {
			queues[vol[j]].push(j);
		}
	}

	// flood
	vigra::UInt8 grayLevel = 0;
	for(;;) {
		while(!queues[grayLevel].empty()) {
			// label pixel and remove from queue
			vigra::MultiArrayIndex j = queues[grayLevel].front();
			queues[grayLevel].pop();

			// add unlabeled neighbors to queues
			// left, upper, and front voxel
			typename vigra::MultiArrayView<3, vigra::UInt32, ST>::difference_type coordinate = labeling.scanOrderIndexToCoordinate(j);
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] != 0) {
					--coordinate[d];
					if(labeling[coordinate] == 0) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
						vigra::UInt8 queueIndex = (vigra::UInt8) std::max(vol[coordinate], grayLevel) * factors[labeling[j]];
						labeling[k] = labeling[j]; // label pixel
						
						queues[queueIndex].push(k);
					}
					++coordinate[d];
				}
			}
			// right, lower, and back voxel
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] < labeling.shape(d)-1) {
					++coordinate[d];
					if(labeling[coordinate] == 0) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
            vigra::UInt8 queueIndex = (vigra::UInt8) std::max(vol[coordinate], grayLevel) * factors[labeling[j]];
						labeling[k] = labeling[j]; // label pixel
						queues[queueIndex].push(k);
					}
					--coordinate[d];
				}
			}
		}
		if(grayLevel == 255) {
			break;
		}
		else {
			queues[grayLevel] = std::queue<vigra::MultiArrayIndex>(); // free memory
			++grayLevel;
		}
	}
	int t2 = clock();
	
	double time = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;
	
	printf("tws took: %f\n", time);
}

template<class T, class ST>
void twsc
(
	const vigra::MultiArrayView<3, vigra::UInt8, ST>& vol,
	vigra::MultiArrayView<3, T, ST>& labeling, 
	vigra::MultiArrayView<3, vigra::UInt8, ST>& directions,
	std::map<std::pair<T, T>, std::pair<vigra::MultiArrayIndex, vigra::MultiArrayIndex> >& adjacency
)
{
	// define 256 queues, one for each gray level.
	std::vector<std::queue<vigra::MultiArrayIndex> > queues(256);

	// add each unlabeled pixels which is adjacent to a seed
	// to the queue corresponding to its gray level
	for(vigra::MultiArrayIndex j = 0; j < labeling.size(); ++j) {
		if(isAtSeedBorder<T>(labeling, j)) {
			queues[vol[j]].push(j);
		}
	}

	// flood
	vigra::UInt8 grayLevel = 0;
	for(;;) {
		while(!queues[grayLevel].empty()) {
			// label pixel and remove from queue
			vigra::MultiArrayIndex j = queues[grayLevel].front();
			queues[grayLevel].pop();

			// add unlabeled neighbors to queues
			// left, upper, and front voxel
			typename vigra::MultiArrayView<3, vigra::UInt32, ST>::difference_type coordinate = labeling.scanOrderIndexToCoordinate(j);
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] != 0) {
					--coordinate[d];
					if(labeling[coordinate] == 0) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
						vigra::UInt8 queueIndex = std::max(vol[coordinate], grayLevel);
						labeling[k] = labeling[j]; // label pixel
						directions[k] = d+1; // save direction
						queues[queueIndex].push(k);
					}
					else if(labeling[coordinate] != labeling[j]) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
						if(labeling[j] < labeling[k]) {
							std::pair<T, T> p(labeling[j], labeling[k]);
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<vigra::MultiArrayIndex, vigra::MultiArrayIndex>(j, k);
							}
						}
						else {
							std::pair<T, T> p(labeling[k], labeling[j]);
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<vigra::MultiArrayIndex, vigra::MultiArrayIndex>(k, j);
							}
						}
					}
					++coordinate[d];
				}
			}
			
			// right, lower, and back voxel
			for(unsigned short d = 0; d<3; ++d) {
				if(coordinate[d] < labeling.shape(d)-1) {
					++coordinate[d];
					if(labeling[coordinate] == 0) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
						vigra::UInt8 queueIndex = std::max(vol[coordinate], grayLevel);
						labeling[k] = labeling[j]; // label pixel
						directions[k] = d+4; // save direction
						queues[queueIndex].push(k);
					}
					else if(labeling[coordinate] != labeling[j]) {
						vigra::MultiArrayIndex k = labeling.coordinateToScanOrderIndex(coordinate);
						if(labeling[j] < labeling[k]) {
							std::pair<T, T> p(labeling[j], labeling[k]);
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<vigra::MultiArrayIndex, vigra::MultiArrayIndex>(j, k);
							}
						}
						else {
							std::pair<T, T> p(labeling[k], labeling[j]);
							if(adjacency.count(p) == 0) {
								adjacency[p] = std::pair<vigra::MultiArrayIndex, vigra::MultiArrayIndex>(k, j);
							}
						}
					}
					--coordinate[d];
				}
			}
		}
		if(grayLevel == 255) {
			break;
		}
		else {
			queues[grayLevel] = std::queue<vigra::MultiArrayIndex>(); // free memory
			++grayLevel;
		}
	}
}

template<class T, class ST>
inline bool isAtSeedBorder
(
	const vigra::MultiArrayView<3, T, ST>& labeling,
	const vigra::MultiArrayIndex& index
)
{
	if(labeling[index] == 0) {	
		return false; // not a seed voxel
	}
	else {
		typename vigra::MultiArrayView<3, vigra::UInt32, ST>::difference_type coordinate
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

#endif //TWS_HXX
