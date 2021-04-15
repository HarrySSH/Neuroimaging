%% List of Files for Harry for ML version of Connectome Reconstruction

% IMPORTANT NOTE 1: Regions 12 and 225 (same region in the cerebellum, in
% opposite hemispheres) are 0 in gene expression and cell type maps. We have
% therefore removed these cerebellar regions entirely from all atlases for
% this package. Hence all networks are 424 x 424 instead of 426 x 426.

% IMPORTANT NOTE 2: All cell type and region orderings across all variables
% and .mat files are consistent. For the region names cell array, it is
% only 212 instead of 424 long--this is because the region names are the
% same in both hemispheres.

%1. Connectome (Directed/'C_dir' & Undirected/'C_sym'), total connectivity 
% between region pairs, 424 x 424 matrices. Units are arbitrary. Saved in 
% 'Connectomes.mat'. Matrix is organized such that C_dir(1,2) indicates the
% connection going from Region-1 --> Region-2 and C_dir(2,1) indicates the
% connection going from Region-2 --> Region-1. Overall, the connectome is
% organized going down the rows and across the columns such that the first
% 212 regions are the RH and the next 212 regions are the same ones in the
% same order in the LH.

%2. Region volumes, in a 424 vector, to get connectivity density, divide
% each row in connectomes by each entry in the vector to get density. Units
% are in 200 micron per vertex voxels.

%3. String names of Regions, 424 cell array (3 columns, for different
% resolutions). Saved in 'region_names' in file 'Region_Names.mat'.

%4. Cell Type Map (Work in Progress, just for experimenting with code), 424
% region X 25 cell type matrix of mean voxel per cell type per region.
% Units are arbitrary. Saved in 'cellmaps' in file 'CellType_Maps.mat'.
% Important to note that scaling between columns is also arbitrary, so
% columns should be normalized, we suggest minmax or L1.

%5. Matrix of Region -- Region straight line distances, between regional
% geometric centers, 424 x 424, symmetric. Saved in variable, 'D', in file
% 'Interregional_Distances.mat' and units are arbitrary.

%6. Cell Type Names, 25 element cell array of strings. Saved in 'cellnames'
% in file 'CellType_Names.mat'

%7. 3D matrix in Raj Lab listBmap space, 424 unique integer IDs, 
% 1 per voxel. Voxels are 200 microns per vertex. Saved in 'brainat-3d' in
% the file 'MouseBrain_3DAtlas.mat'.

%8. Gene Expression atlas, 424 regions x 4083 genes, ISH probe intensity
% per gene per region. Scale between gene columns is arbitrary

%9. Gene names, for each of the 4083 genes, 4083 element string cell array