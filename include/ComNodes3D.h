/***************************************************************************
  ComNodes.h  -  Library to manage communication of field values among processors
  -------------------
begin                : May 2008
copyright            : KUL Leuven
developers           : Stefano Markidis, Giovanni Lapenta

 ***************************************************************************/

#ifndef ComNodes3D_H
#define ComNodes_H

#include "ComBasic3D.h"
#include "TimeTasks.h"

extern TimeTasks timeTasks;

// boundary condition for fields
#include "BcFields3D.h"

/** communicate ghost cells (FOR NODES) */
void communicateNode(int nx, int ny, int nz, double ***vector, VirtualTopology3D * vct);

/** communicate ghost cells (FOR NODES) */
void communicateNodeBC(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

/** communicate ghost cells (FOR NODES) with particles BC*/

//! Not used
void communicateNodeBC_P(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

/** SPECIES: communicate ghost cells */
void communicateNode(int nx, int ny, int nz, double ****vector, int ns, VirtualTopology3D * vct);

//! PARTICLES
//! SPECIES: communicate ghost cells !//
void communicateNode_P(int nx, int ny, int nz, double ****vector, int ns, VirtualTopology3D * vct);

// 
/** communicate ghost cells (FOR CENTERS) */
void communicateCenter(int nx, int ny, int nz, double ***vector, VirtualTopology3D * vct);

//? communicate ghost cells (FOR CENTERS) with BOX stencil -- used in communicateCenterBoxStencilBC ?//
void communicateCenterBoxStencilBC(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

//! Particles
//? communicate ghost cells (FOR CENTERS) with BOX stencil -- used in EMfields3D::smooth ?//
void communicateCenterBoxStencilBC_P(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

//? used in EMfields3D::smooth ?//
void communicateNodeBoxStencilBC_P(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

//? used in EMfields3D::smoothE ?//
void communicateNodeBoxStencilBC(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

//! SPECIES: communicate ghost cells !//
void communicateCenter(int nx, int ny, int nz, double ****vector, int ns, VirtualTopology3D * vct);

//? /////////// communication + BC ////////////////////////// ?//
void communicateCenterBC(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

//? /////////// communication + BC ////////////////////////// ?//
void communicateCenterBC_P(int nx, int ny, int nz, double ***vector, int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, VirtualTopology3D * vct);

#endif
