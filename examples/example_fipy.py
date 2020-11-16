# This source code is based on the Stokes cavity example of FiPy:
# https://www.ctcms.nist.gov/fipy/examples/flow/generated/examples.flow.stokesCavity.html

# ------------
# FiPy license
# ------------

# This software was developed by employees of the `National Institute of
# Standards and Technology`_ (NIST_), an agency of the Federal Government and
# is being made available as a public service.  Pursuant to `title 17 United
# States Code Section 105`_, works of NIST_ employees are not subject to
# copyright protection in the United States.  This software may be subject to
# foreign copyright.  Permission in the United States and in foreign
# countries, to the extent that NIST_ may hold copyright, to use, copy,
# modify, create derivative works, and distribute this software and its
# documentation without fee is hereby granted on a non-exclusive basis,
# provided that this notice and disclaimer of warranty appears in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
# EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY
# WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL
# CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR
# FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT
# LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT
# OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR
# NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT
# INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR
# NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE
# SOFTWARE OR SERVICES PROVIDED HEREUNDER.

# .. _National Institute of Standards and Technology: http://www.nist.gov/
# .. _NIST: http://www.nist.gov/
# .. _title 17 United States Code Section 105: https://www.copyright.gov/title17/92chap1.html#105

from skfem.visuals.matplotlib import *

from adaptmesh import triangulate

dom = [
    (0.7, -0.45),
    (0.8165467625899281, -0.43571428571428594),
    (0.9388489208633093, -0.4035714285714287),
    (1.064748201438849, -0.36071428571428577),
    (1.1906474820143886, -0.3142857142857145),
    (1.2877697841726619, -0.25714285714285734),
    (1.356115107913669, -0.1892857142857145),
    (1.3956834532374103, -0.12142857142857144),
    (1.406474820143885, -0.03571428571428559),
    (1.4028776978417268, 0.04285714285714293),
    (1.3741007194244608, 0.11071428571428554),
    (1.3309352517985613, 0.18571428571428572),
    (1.2517985611510793, 0.2607142857142859),
    (1.158273381294964, 0.30714285714285716),
    (1.0467625899280577, 0.3500000000000001),
    (0.9280575539568345, 0.3857142857142859),
    (0.7985611510791368, 0.4142857142857146),
    (0.7, 0.43),
]

dom = [(xs[0] - 0.7, xs[1]) for xs in dom]

m = triangulate(dom, quality=0.92)

draw(m)
show()


from fipy.meshes.mesh2D import Mesh2D
from skfem import MappingAffine

t = m.t.copy()
t2f = m.t2f.copy()
mapping = MappingAffine(m)
tmpt = t[:, mapping.detA < 0]
tmpt[[0, 1]] = tmpt[[1, 0]]
t[:, mapping.detA < 0] = tmpt
tmpt = t2f[:, mapping.detA > 0]
tmpt[[0, 1]] = tmpt[[1, 0]]
t2f[:, mapping.detA > 0] = tmpt

# recreate facets with orientation
facets = np.hstack(
    (
        t[[0, 1]],
        t[[1, 2]],
        t[[2, 0]],
    )
)

m.refine()

mesh = Mesh2D(m.p, m.facets, m.t2f)


from fipy import CellVariable, DiffusionTerm, FaceVariable, Grid2D, Viewer
from fipy.tools import numerix

viscosity = 100
U = 1.0
pressureRelaxation = 0.8
velocityRelaxation = 0.5


pressure = CellVariable(mesh=mesh, name="pressure")
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name="X velocity")
yVelocity = CellVariable(mesh=mesh, name="Y velocity")


velocity = FaceVariable(mesh=mesh, rank=1)

xVelocityEq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.0, 0.0])
yVelocityEq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.0, 1.0])


ap = CellVariable(mesh=mesh, value=1.0)
coeff = 1.0 / ap.arithmeticFaceValue * mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = DiffusionTerm(coeff=coeff) - velocity.divergence


from fipy.variables.faceGradVariable import _FaceGradVariable

volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name="Volume")
contrvolume = volume.arithmeticFaceValue


X, Y = mesh.faceCenters

yVelocity.constrain(0.0, mesh.exteriorFaces)
xVelocity.constrain(0.0, mesh.exteriorFaces)
yVelocity.constrain(-10, mesh.facesLeft)
pressureCorrection.constrain(0.0, mesh.facesRight & (Y < 0.1))

from builtins import range

import fipy as fp


for sweep in range(300):
    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix
    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)
    ## update the ap coefficient from the matrix diagonal
    ap[:] = -numerix.asarray(xmat.takeDiagonal())
    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)
    velocity[0] = (
        xVelocity.arithmeticFaceValue
        + contrvolume
        / ap.arithmeticFaceValue
        * (presgrad[0].arithmeticFaceValue - facepresgrad[0])
    )
    velocity[1] = (
        yVelocity.arithmeticFaceValue
        + contrvolume
        / ap.arithmeticFaceValue
        * (presgrad[1].arithmeticFaceValue - facepresgrad[1])
    )
    velocity[..., mesh.exteriorFaces.value] = 0.0
    velocity[0, mesh.facesTop.value] = U
    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector
    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection)
    ## update the velocity using the corrected pressure
    xVelocity.setValue(
        xVelocity - pressureCorrection.grad[0] / ap * mesh.cellVolumes
    )
    yVelocity.setValue(
        yVelocity - pressureCorrection.grad[1] / ap * mesh.cellVolumes
    )
    if __name__ == "__main__":
        if sweep % 10 == 0:
            print(
                "sweep:",
                sweep,
                ", x residual:",
                xres,
                ", y residual",
                yres,
                ", p residual:",
                pres,
                ", continuity:",
                max(abs(rhs)),
            )

from skfem import ElementTriP0, ElementTriP1, InteriorBasis, project

ib = InteriorBasis(m, ElementTriP0())
plot(ib, pressure.value)
ib2 = InteriorBasis(m, ElementTriP1())
velo = project(
    (xVelocity.value) ** 2 + (yVelocity.value) ** 2,
    basis_from=ib,
    basis_to=ib2,
)
plot(ib2, velo, cmap="magma", shading="gouraud")

import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver

m = MeshQuad()

import numpy as np

X, Y = np.meshgrid(np.linspace(0.05, 0.7, 10), np.linspace(-0.45, 0.45, 10))
interpx = ib.interpolator(xVelocity.value)
interpy = ib.interpolator(yVelocity.value)
inputp = np.vstack((X.flatten(), Y.flatten()))
plt.streamplot(
    X,
    Y,
    interpx(inputp).reshape(X.shape),
    interpy(inputp).reshape(X.shape),
    color="w",
)

show()
