# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module provides utility functions for the models package
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import math
import textwrap

from collections import deque

import numpy as np

from ..extern.six.moves import xrange, zip_longest


__all__ = ['ExpressionTree', 'check_broadcast', 'poly_map_domain', 'comb']


class ExpressionTree(object):
    __slots__ = ['left', 'right', 'value']

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        return self.format_tree_ascii()

    @property
    def isleaf(self):
        return self.left is None and self.right is None

    def traverse_preorder(self):
        stack = deque([self])
        while stack:
            node = stack.pop()
            yield node

            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    def traverse_inorder(self):
        stack = deque()
        node = self
        while stack or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node
                node = node.right

    def traverse_postorder(self):
        stack = deque()
        node = self
        last = None
        while stack or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                parent = stack[-1]
                if parent.right is not None and last is not parent.right:
                    node = parent.right
                else:
                    stack.pop()
                    yield parent
                    last = parent

    def evaluate(self, operators, getter=None):
        """Evaluate the expression represented by this tree.

        ``Operators`` should be a dictionary mapping operator names ('tensor',
        'product', etc.) to a function that implements that operator for the
        correct number of operands.

        If given, ``getter`` is a function evaluated on each *leaf* node's
        value before applying the operator between them.  This could be used,
        for example, to operate on an attribute of the node values rather than
        directly on the node values.
        """

        operands = deque()

        for node in self.traverse_postorder():
            if node.isleaf:
                # For a "tree" containing just a single operator at the root
                operands.append(getter(node.value))
            else:
                operator = operators[node.value]
                right = operands.pop()
                left = operands.pop()
                operands.append(operator(left, right))

        return operands.pop()

    def copy(self):
        # Hopefully this won't blow the stack for any practical case; if such a
        # case arises that this won't work then I suppose we can find an
        # iterative approach.

        children = []
        for child in (self.left, self.right):
            if isinstance(child, ExpressionTree):
                children.append(child.copy())
            else:
                children.append(child)

        return self.__class__(self.value, left=children[0], right=children[1])

    def format_expression(self, operator_precedence, format_leaf=None):
        leaf_idx = 0
        operands = deque()

        if format_leaf is None:
            format_leaf = lambda i, l: '[{0}]'.format(i)

        for node in self.traverse_postorder():
            if node.isleaf:
                operands.append(format_leaf(leaf_idx, node))
                leaf_idx += 1
                continue

            oper_order = operator_precedence[node.value]
            right = operands.pop()
            left = operands.pop()

            if (node.left is not None and not node.left.isleaf and
                    operator_precedence[node.left.value] < oper_order):
                left = '({0})'.format(left)
            if (node.right is not None and not node.right.isleaf and
                    operator_precedence[node.right.value] < oper_order):
                right = '({0})'.format(right)

            operands.append(' '.join((left, node.value, right)))

        return ''.join(operands)

    # TODO: This could still use a lot of improvement; in particular the trees
    # it outputs are often too wide, and could be made more compactly.  More
    # formatting control would be useful too.
    def format_tree_ascii(self, format_leaf=None):
        """
        Format the tree using an ASCII character representation.

        Parameters
        ----------
        format_leaf : callable
            A function of a single argument which, given a node value,
            returns a string representing that node in the tree display.
        """

        stack = deque()
        leaf_idx = 0

        if format_leaf is None:
            format_leaf = lambda i, l: '[{0}]'.format(i)

        for node in self.traverse_postorder():
            if node.isleaf:
                text = format_leaf(leaf_idx, node.value)
                width = len(text)
                anchor = int(math.ceil(width / 2))
                stack.append((width, anchor, [text]))
                leaf_idx += 1
                continue

            right_width, right_anchor, right = stack.pop()
            left_width, left_anchor, left = stack.pop()

            #if left_width > right_width:
            #    right = [r.center(left_width) for r in right]
            #    child_width = left_width
            #elif right_width > left_width:
            #    left = [l.center(right_width) for l in left]
            #    child_width = right_width
            #else:
            #    child_width = left_width  # without loss of generality

            #root = '[{0}]'.format(node.value)
            #spine = '/   \\'
            ##fill = ' ' * len(spine)
            #fill = ' ' * 7
            ##width = 2 * child_width + len(fill)
            #width = left_width + len(fill) + right_width
            root = '[{0}]'.format(node.value)

            # Here we deterine the optimal spacing between the left and right
            # subtrees.  To begin with there is at least *one* space.  Then the
            # distance is calculated between the anchor point on the left
            # subtree and the right subtree.
            # This distance must be at *least* with width of the root node plus
            # two (for the diagonal edges coming off the root)  If not,
            # additional spaces are added until they are.  If it is already
            # more than that then the edges are extended to reach the anchors
            min_dist = len(root) + 2
            dist = (left_width - left_anchor) + right_anchor

            if dist < min_dist:
                fill = ' ' * (min_dist - dist + 1)
            else:
                fill = ' '

            dist = max(dist, min_dist)
            root_offset = left_anchor + (dist // 2) - (len(root) // 2)

            spine_left = (' ' * left_anchor +
                          '_' * (root_offset - left_anchor - 1) + '/')
            import sys
            sys.stderr.write(str(right_anchor - len(root) // 2 - len(fill) - 1) + '\n')
            spine_right = '\\' + '_' * (dist - len(root) - right_anchor - 1)

            spine = spine_left + ' ' * len(root) + spine_right

            lines = [' ' * root_offset + root, spine]

            #spine_left = '_' * left_anchor + '/'
            #spine_right = '\\' + '_' * right_anchor
            #root_center = math.ceil(len(root) / 2)
            #left_offset = left_width - left_anchor

            #spine = spine_left + ' ' * len(root) + spine_right

            #offset = child_width - (child_width % 2)
            #spine = spine.center(width - 2 - offset, '_').center(width)
            #root = root.center(width)
            #fill = ' ' * (left_offset + right_offset + len(root))
            #fill = ' ' * len(root)

            #lines = [' ' * left_width + root,
            #         ' ' * left_offset + spine]

            if len(right) > len(left):
                fillvalue = ' ' * left_width
            else:
                fillvalue = ''

            for l, r in zip_longest(left, right, fillvalue=fillvalue):
                lines.append(l.ljust(left_width) + fill + r)

            width = left_width + len(fill) + right_width
            #anchor = left_width + root_center
            anchor = root_offset + math.ceil(len(root) / 2)
            import sys
            sys.stderr.write('{0}:\n'.format(node.value))
            sys.stderr.write('{0} {1}\n'.format(width, anchor))
            stack.append((width, anchor, lines))

        return textwrap.dedent('\n'.join(stack[0][2]))


class IncompatibleShapeError(ValueError):
    def __init__(self, shape_a, shape_a_idx, shape_b, shape_b_idx):
        super(IncompatibleShapeError, self).__init__(
                shape_a, shape_a_idx, shape_b, shape_b_idx)


def check_broadcast(*shapes):
    """
    Determines whether two or more Numpy arrays can be broadcast with each
    other based on their shape tuple alone.

    Parameters
    ----------
    *shapes : tuple
        All shapes to include in the comparison.  If only one shape is given it
        is passed through unmodified.  If no shapes are given returns an empty
        `tuple`.

    Returns
    -------
    broadcast : `tuple`
        If all shapes are mutually broadcastable, returns a tuple of the full
        broadcast shape.
    """

    if len(shapes) == 0:
        return ()
    elif len(shapes) == 1:
        return shapes[0]

    reversed_shapes = (reversed(shape) for shape in shapes)

    full_shape = []

    for dims in zip_longest(*reversed_shapes, fillvalue=1):
        max_dim = 1
        max_dim_idx = None
        for idx, dim in enumerate(dims):
            if dim == 1:
                continue

            if max_dim == 1:
                # The first dimension of size greater than 1
                max_dim = dim
                max_dim_idx = idx
            elif dim != max_dim:
                raise IncompatibleShapeError(
                    shapes[max_dim_idx], max_dim_idx, shapes[idx], idx)

        full_shape.append(max_dim)

    return tuple(full_shape[::-1])


def poly_map_domain(oldx, domain, window):
    """
    Map domain into window by shifting and scaling.

    Parameters
    ----------
    oldx : array
          original coordinates
    domain : list or tuple of length 2
          function domain
    window : list or tuple of length 2
          range into which to map the domain
    """
    domain = np.array(domain, dtype=np.float64)
    window = np.array(window, dtype=np.float64)
    scl = (window[1] - window[0]) / (domain[1] - domain[0])
    off = (window[0] * domain[1] - window[1] * domain[0]) / (domain[1] - domain[0])
    return off + scl * oldx


def comb(N, k):
    """
    The number of combinations of N things taken k at a time.

    Parameters
    ----------
    N : int, array
        Number of things.
    k : int, array
        Number of elements taken.

    """
    if (k > N) or (N < 0) or (k < 0):
        return 0
    val = 1
    for j in xrange(min(k, N - k)):
        val = (val * (N - j)) / (j + 1)
    return val


def array_repr_oneline(array):
    """
    Represents a multi-dimensional Numpy array flattened onto a single line.
    """

    r = np.array2string(array, separator=',', suppress_small=True)
    return ' '.join(l.strip() for l in r.splitlines())
