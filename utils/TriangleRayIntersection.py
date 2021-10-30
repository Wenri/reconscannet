import numpy as np
import torch


def TriangleRayIntersection(orig, dir, vert0, vert1, vert2):
    """TRIANGLERAYINTERSECTION Ray/triangle intersection.
       INTERSECT = TriangleRayIntersection(ORIG, DIR, VERT1, VERT2, VERT3)
         calculates ray/triangle intersections using the algorithm proposed
         BY Möller and Trumbore (1997), implemented as highly vectorized
         MATLAB code. The ray starts at ORIG and points toward DIR. The
         triangle is defined by vertix points: VERT1, VERT2, VERT3. All input
         arrays are in Nx3 or 1x3 format, where N is number of triangles or
         rays.

      [INTERSECT, T, U, V, XCOOR] = TriangleRayIntersection(...)
        Returns:
        * Intersect - boolean array of length N informing which line and
                    triangle pair intersect
        * t   - distance from the ray origin to the intersection point in
                units of |dir|. Provided only for line/triangle pair that
                intersect unless 'fullReturn' parameter is true.
        * u,v - barycentric coordinates of the intersection point
        * xcoor - carthesian coordinates of the intersection point

      TriangleRayIntersection(...,'param','value','param','value'...) allows
       additional param/value pairs to be used. Allowed parameters:
       * planeType - 'one sided' or 'two sided' (default) - how to treat
           triangles. In 'one sided' version only intersections in single
           direction are counted and intersections with back facing
              tringles are ignored
       * lineType - 'ray' (default), 'line' or 'segment' - how to treat rays:
           - 'line' means infinite (on both sides) line;
           - 'ray' means infinite (on one side) ray comming out of origin;
           - 'segment' means line segment bounded on both sides
       * border - controls border handling:
           - 'normal'(default) border - triangle is exactly as defined.
              Intersections with border points can be easily lost due to
              rounding errors.
           - 'inclusive' border - triangle is marginally larger.
              Intersections with border points are always captured but can
              lead to double counting when working with surfaces.
           - 'exclusive' border - triangle is marginally smaller.
              Intersections with border points are not captured and can
              lead to under-counting when working with surfaces.
       * epsilon - (default = 1e-5) controls border size
       * fullReturn - (default = false) controls returned variables t, u, v,
           and xcoor. By default in order to save time, not all t, u & v are
           calculated, only t, u & v for intersections can be expected.
           fullReturn set to true will force the calculation of them all.

    ALGORITHM:
     Function solves
           |t|
       M * |u| = (o-v0)
           |v|
     for [t; u; v] where M = [-d, v1-v0, v2-v0]. u,v are barycentric coordinates
     and t - the distance from the ray origin in |d| units
     ray/triangle intersect if u>=0, v>=0 and u+v<=1

    NOTE:
     The algorithm is able to solve several types of problems:
     * many faces / single ray  intersection
     * one  face  / many   rays intersection
     * one  face  / one    ray  intersection
     * many faces / many   rays intersection
     In order to allow that to happen all imput arrays are expected in Nx3
     format, where N is number of vertices or rays. In most cases number of
     vertices is different than number of rays, so one of the imputs will
     have to be cloned to have the right size. Use "repmat(A,size(B,1),1)".

    Based on:
     *"Fast, minimum storage ray-triangle intersection". Tomas Möller and
       Ben Trumbore. Journal of Graphics Tools, 2(1):21--28, 1997.
       http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
     * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
     * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/raytri.c

    Author:
       Jarek Tuszynski (jaroslaw.w.tuszynski@leidos.com)

    License: BSD license (http://en.wikipedia.org/wiki/BSD_licenses)

    # In case of single points clone them to the same size as the rest
    N = max(concat([size(orig, 1), size(dir, 1), size(vert0, 1), size(vert1, 1), size(vert2, 1)]))
    if size(orig, 1) == 1 and N > 1 and size(orig, 2) == 3:
        orig = repmat(orig, N, 1)
    if size(dir, 1) == 1 and N > 1 and size(dir, 2) == 3:
        dir = repmat(dir, N, 1)
    if size(vert0, 1) == 1 and N > 1 and size(vert0, 2) == 3:
        vert0 = repmat(vert0, N, 1)
    if size(vert1, 1) == 1 and N > 1 and size(vert1, 2) == 3:
        vert1 = repmat(vert1, N, 1)
    if size(vert2, 1) == 1 and N > 1 and size(vert2, 2) == 3:
        vert2 = repmat(vert2, N, 1)
    """

    # Check if all the sizes match
    assert orig.shape == vert0.shape and orig.shape == vert1.shape and orig.shape == vert2.shape and \
           orig.shape == dir.shape and orig.shape[1] == 3, 'All input vectors have to be in Nx3 format.'

    # Read user preferences
    eps = 1e-05
    planeType = 'one sided'
    lineType = 'ray'
    border = 'normal'
    fullReturn = False

    # Set up border parameter
    zero = {
        'normal': 0.0,
        'inclusive': np.finfo(float).eps,
        'exclusive': -np.finfo(float).eps
    }[border]

    # Find faces parallel to the ray
    edge1 = vert1 - vert0
    edge2 = vert2 - vert0
    tvec = orig - vert0

    pvec = torch.cross(dir, edge2)

    det = torch.sum(torch.multiply(edge1, pvec), dim=-1)

    if 'two sided' == planeType:
        angleOK = torch.abs(det) > eps
    else:
        if 'one sided' == planeType:
            angleOK = det > eps
        else:
            raise ValueError('Triangle parameter must be either "one sided" or "two sided"')

    if not torch.any(angleOK):
        return angleOK

    # Different behavior depending on one or two sided triangles
    det[torch.logical_not(angleOK)] = np.nan

    u = torch.sum(torch.multiply(tvec, pvec), dim=-1) / det

    if fullReturn:
        # calculate all variables for all line/triangle pairs
        qvec = torch.cross(tvec, edge1)
        v = torch.sum(torch.multiply(dir, qvec), dim=-1) / det
        t = torch.sum(torch.multiply(edge2, qvec), dim=-1) / det
        # test if line/plane intersection is within the triangle
        ok = torch.logical_and(angleOK, torch.logical_and(
            torch.logical_and(u >= - zero, v >= - zero), u + v <= 1.0 + zero))
    else:
        # limit some calculations only to line/triangle pairs where it makes
        # a difference. It is tempting to try to push this concept of
        # limiting the number of calculations to only the necessary to "u"
        # and "t" but that produces slower code
        v = torch.full_like(u, fill_value=np.nan)
        t = torch.full_like(u, fill_value=np.nan)
        ok = torch.logical_and(angleOK, torch.logical_and(u >= - zero, u <= 1.0 + zero))
        # if all line/plane intersections are outside the triangle than no intersections
        if not torch.any(ok):
            return ok
        qvec = torch.cross(tvec[ok, :], edge1[ok, :])
        v[ok] = torch.sum(torch.multiply(dir[ok, :], qvec), dim=-1) / det[ok]
        if lineType != 'line':
            t[ok] = torch.sum(torch.multiply(edge2[ok, :], qvec), dim=-1) / det[ok]
        # test if line/plane intersection is within the triangle
        ok = torch.logical_and(ok, torch.logical_and(v >= - zero, u + v <= 1.0 + zero))

    # Test where along the line the line/plane intersection occurs
    if 'line' == lineType:
        intersect = ok
    else:
        if 'ray' == lineType:
            intersect = torch.logical_and(ok, t >= - zero)
        else:
            if 'segment' == lineType:
                intersect = torch.logical_and(ok, torch.logical_and(t >= - zero, t <= 1.0 + zero))
            else:
                raise ValueError('lineType parameter must be either "line", "ray" or "segment"')

    # calculate intersection coordinates if requested
    # if (nargout > 4):
    #     ok = logical_or(intersect, fullReturn)
    #     xcoor[ok, :] = vert0[ok, :] + np.multiply(edge1[ok, :], repmat(u(ok, 1), 1, 3)) + np.multiply(
    #         edge2[ok, :], repmat(v[ok, 1], 1, 3))

    return intersect
