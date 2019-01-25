import sys
import json
import vtk

colors = vtk.vtkNamedColors()

class Point:
    def __init__(self, x=0, y=0, z=0, t="H"):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def __str__(self):
        return "(" + ", ".join([str(i) for i in [self.x, self.y, self.z, self.t]]) + ")"

def getPoints(filename):
    with open(filename) as fp:
        data = json.load(fp)

    points = []

    coords = data["coords"].strip().split(" ")
    hp = list(data["hpchain"])

    for coord in coords:
        nums = coord.strip("()").split(",")
        nums = [int(i) for i in nums]
        points.append(Point(nums[0], nums[1], nums[2]))

    for p, h in zip(points, hp):
        p.t = h

    return points

def sphereActor(x, y, z, color="blue"):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(x, y, z)
    sphereSource.SetRadius(0.2)
    sphereSource.SetPhiResolution(100)
    sphereSource.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))

    return actor

def lineActor(p1, p2):
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(p1.x, p1.y, p1.z)
    lineSource.SetPoint2(p2.x, p2.y, p2.z)

    tubeSource = vtk.vtkTubeFilter()
    tubeSource.SetInputConnection(lineSource.GetOutputPort())
    tubeSource.SetRadius(0.25/4)
    tubeSource.SetNumberOfSides(100)
    tubeSource.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeSource.GetOutputPort())
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(colors.GetColor3d("black"))
    actor.SetMapper(mapper)

    return actor

def main():
    if len(sys.argv) == 1:
        print("Usage: {} [input file]\n".format(sys.argv[0]))
        return 1
    else:
        points = getPoints(sys.argv[1])

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetWindowName("Sphere")
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.SetBackground(colors.GetColor3d("white"))

    # Render first bead
    p = points[0]
    renderer.AddActor(sphereActor(p.x, p.y, p.z, "cyan" if p.t is "P" else "orange"))

    # Render beads
    for p in points[1:]:
        renderer.AddActor(sphereActor(p.x, p.y, p.z, "blue" if p.t is "P" else "red"))

    # Render bonds
    for i in range(1, len(points)):
        renderer.AddActor(lineActor(points[i-1], points[i]))

    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == '__main__':
    main()
