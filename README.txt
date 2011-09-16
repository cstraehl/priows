This is a python wrapper and c++ code that implements an interactive segmentation algorithm.
the algorithm (a straight forward watershed extension) is published in:
    
    Carving: Scalable Interactive Segmentation of Neural Volume Electron Microscopy Images,
    Christoph-N. Straehle, Ullrich Kothe, Graham Knott,and Fred A. Hamprecht, MICCAI 2011


Installation
============
The build system is based on cmake.
goto the source directory and type

    > cmake .
    > make
    > make install

to compile and install the python module. 

For compiling the following
software packages are needed:

Software Requirements:
---------------------
    CMake 2.6
    vigra 1.8
    vigranumpy
    boost-python
    

Usage
=====
after installation the new vigra python module "vigra.priows" is available.

    
    #
    # Example script utilizing the vigra.priows python module
    #
    
    import vigra, numpy
    import vigra.priows
    
    shape  = (20,30,40)
    
    # create some 3d data
    dataVol = numpy.ndarray(shape, numpy.uint8)
    
    
    # create some 3d segmentation seed data
    seedVol = numpy.zeros(shape, numpy.uint8)
    seedVol[1,1,1] = 1
    seedVol[18,28,38] = 2
    
    
    # get the flattened indices and labels of the seeds
    # NOTE: this is a crucial step for an interacitve segmentation application
    #       you should not use numpy nonzero on the complete seed volume, this
    #       would destroy the whole point of the problem reduction due to supervoxels.
    #       instead manage your seeds in a sparse way from the beginning ...
    seedIndices = numpy.nonzero(seedVol.ravel())[0]
    seedLabels = seedVol.ravel()[seedIndices]
    
    
    seg = vigra.priows.segmentor(dataVol, useDifference = False,min = 0, max = 255, queuecount = 255, dontUseSuperVoxels=False )


    #we can retrieve the supervoxel assignments of the volume
    volumeSupervoxels = seg.getVolumeBasins()    
    
    seg.setSeeds(seedLabels, seedIndices.astype(numpy.uint32))
    
    
    # now we can execute the priorized supervoxel watershed
    supervoxelLabels = seg.doWS(bias = 0.9, biasThreshold=64, biasedLabel = 1, virtualBackgroundSeeds=False)
    
    
    # and can retrieve the label assignment of a region of interest
    segmentation = supervoxelLabels[volumeSupervoxels[:,2,:]]
    
    print segmentation
    
    
Directory Layout:
=======================

superVoxelSeg.hxx : the c++ implementation of the algorithm
python_svs.cpp    : python wrapper
CMakeList.txt     : cmake build system configuration file
CMakeFiles        : some custom CMake scripts for finding e.g. vigra