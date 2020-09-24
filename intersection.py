import shapely
from shapely.geometry import LineString, Point

# [(318, 3), (318, 722), (636, 722), (636, 6)]
# [(2, 241), (959, 242), (963, 484), (0, 484)]
A = (318, 3)
B = (318, 722)

#line 2
C = (2, 241)
D = (959, 242)

line1 = LineString([A, B])
line2 = LineString([C, D])

int_pt = line1.intersection(line2)
point_of_intersection = int_pt.x, int_pt.y

print(point_of_intersection)