"""Small additions to the CadQuery namespace our programs run in (training data, rewards, inference).

near(p, ...) - selector for the edge (or face) truly closest to each given point. CadQuery's own
NearestToPointSelector compares object *centres*, so a circular edge (centre = circle centre) can never be picked by
pointing at it; this one measures the real distance from the point to the edge. Several points select several edges:
    r = cq.Workplane('XY').box(100, 60, 20).edges(near((50, 30, 0), (-50, 30, 0))).fillet(5)
"""


def namespace():
    import cadquery as cq

    class Near(cq.Selector):
        def __init__(self, *points):
            self.points = [cq.Vertex.makeVertex(*map(float, p)) for p in points]

        def filter(self, objectList):
            objs = list(objectList)
            if not objs:
                return []
            out = []
            for v in self.points:
                best = min(objs, key=lambda o: o.distance(v))
                if best not in out:
                    out.append(best)
            return out

    def near(*points):
        return Near(*points)

    return {"cq": cq, "near": near}
