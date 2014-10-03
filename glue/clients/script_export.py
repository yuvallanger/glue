from heapq import heappush, heappop, heapify

from ..core.util import as_list
from ..core.state import object_label, disambiguate
from ..core import Client, DataCollection, Data
from . import layer_artist as la

_export_table = {}  # map types to export functions


def unindent(txt):
    lines = txt.splitlines()
    indent = 0
    # strip leading blank lines
    for i, l in enumerate(lines):
        if l:
            lines = lines[i:]
            indent = len(l) - len(l.lstrip())
            break

    return '\n'.join(l[indent:].rstrip() for l in lines)


def export(typ):
    def wrapper(func):
        _export_table[typ] = func
        return func
    return wrapper


def scatter_layer_code(artist, idx, tokens):
    layer = artist.layer
    style = layer.style
    tokens['l%i' % idx] = layer
    tokens['x'] = artist.xatt
    tokens['y'] = artist.yatt

    props = dict(l='{l%i}' % idx,
                 ms=style.markersize,
                 ec=repr(style.color if style.marker == '+' else 'none'),
                 c=repr(style.color),
                 alpha=style.alpha,
                 z=artist.zorder,
                 marker=repr(style.marker))

    block = """
    plt.plot({l}[{{x}}].ravel(), {l}[{{y}}].ravel(),
             markersize={ms}, mec={ec},
             mew=3, mfc={c}, linestyle='None',
             marker={marker},
             alpha={alpha}, zorder={z})
    """.format(**props)
    return unindent(block)


def histogram_code(artist, idx, tokens):
    layer = artist.layer
    style = layer.style
    tokens['l%i' % idx] = layer

    rng = artist.lo, artist.hi
    props = dict(l='{l%i}' % idx,
                 color=repr(style.color),
                 alpha=style.alpha,
                 z=artist.zorder,
                 normed=artist.normed,
                 cumulative=artist.cumulative,
                 ylog=artist.ylog,
                 xlog=artist.xlog,
                 nbins=artist.nbins,
                 rng=rng)

    block = """
    plt.hist({l}[{{att}}].ravel(), bins={nbins}, range={rng},
             normed={normed}, cumulative={cumulative},
             facecolor={color}, alpha={alpha}, zorder={z})
    """
    return unindent(block).format(**props)


@export(Client)
def export_client(client):

    ax = client.axes
    tokens = {}

    code = ['plt.figure()']
    for i, artist in enumerate(client.artists):
        if not (artist.visible and artist.enabled):
            continue

        if isinstance(artist, la.ScatterLayerArtist):
            code.append(scatter_layer_code(artist, i, tokens))
        elif isinstance(artist, la.HistogramLayerArtist):
            code.append(histogram_code(artist, i, tokens))
        else:
            raise ValueError

    code.append(_sync_axes(ax))
    code = '\n'.join(code)

    return Node(code=code, tokens=tokens)


def _sync_axes(ax):

    result = """
    plt.xlabel({xlbl})
    plt.ylabel({ylbl})
    plt.xlim({xlim})
    plt.ylim({ylim})
    plt.xscale({xsc})
    plt.yscale({ysc})
    """.format(xlbl=repr(ax.get_xlabel()),
               ylbl=repr(ax.get_ylabel()),
               xlim=ax.get_xlim(),
               ylim=ax.get_ylim(),
               xsc=repr(ax.get_xscale()),
               ysc=repr(ax.get_yscale()))
    return unindent(result)


@export(DataCollection)
def export_data_collection(dc):
    # test module for now

    result = Node("")
    for data in dc:
        result += Node("{d}=Data(label=%s)" % repr(data.label),
                       gives=[data],
                       tokens=dict(d=data))
        for comp in data.components:
            result += Node("{c}={d}.add_component(%s, label=%s)" %
                           (data[comp].tolist(), repr(comp.label)),
                           gives=[comp],
                           tokens=dict(c=comp, d=data))

    return result


class Node(object):

    def __init__(self, code, needs=None, gives=None, priority=0, tokens=None):
        needs = as_list(needs) if needs is not None else []
        gives = as_list(gives) if gives is not None else []
        tokens = tokens or {}

        self.needs = needs
        self.gives = gives
        self.priority = priority
        self.tokens = tokens
        self.code = code

    @property
    def token_objects(self):
        return list(self.tokens.values())

    def template_fill(self, names):
        tokens = dict((k, names[v]) for k, v in self.tokens.items())
        return self.code.format(**tokens)

    def __add__(self, other):
        return NodeBlock(list(self) + list(other))

    def __str__(self):
        return self.code

    __repr__ = __str__

    def __iter__(self):
        return iter([self])


class NodeBlock(Node):

    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        return iter(self.nodes)

    def __add__(self, other):
        return NodeBlock(list(self) + list(other))

    @property
    def token_objects(self):
        return list(set(sum((n.token_objects for n in self.nodes), [])))

    @property
    def gives(self):
        return sum((n.gives for n in self.nodes), [])

    @property
    def needs(self):
        return sum((n.needs for n in self.nodes), [])

    @property
    def priority(self):
        return min(n.priority for n in self.nodes)

    @property
    def tokens(self):
        pass

    def template_fill(self, names):
        return '\n'.join(n.template_fill(names) for n in self.nodes)


def sorted_nodes(nodes):
    s = [(n.priority, i, n) for i, n in enumerate(nodes) if not n.needs]

    heapify(s)
    todo = [(i, n) for i, n in enumerate(nodes) if n.needs]
    has = set()
    while s:
        _, _, node = heappop(s)
        yield node

        if not node.gives:
            continue

        has.update(node.gives)

        good = (t for t in todo if all(n in has for n in t[1].needs))
        todo = [t for t in todo if not all(n in has for n in t[1].needs)]

        for i, g in good:
            heappush(s, (g.priority, i, g))

    if todo:
        raise RuntimeError("Dependency cycle detected")


def assign_variable_names(nodes):
    tokens = set()
    for n in nodes:
        tokens.update(n.token_objects)
    tokens = list(tokens)

    labels = []
    for tok in tokens:
        labels.append(disambiguate(object_label(tok), labels))

    return dict(zip(tokens, labels))


def template_fill(nodes, names):
    for node in nodes:
        yield node.template_fill(names)


def generate_code(nodes):
    names = assign_variable_names(nodes)
    nodes = sorted_nodes(nodes)
    code = template_fill(nodes, names)
    result = '\n'.join(code)
    try:
        from autopep8 import fix_code
        return fix_code(result)
    except:
        return result


def preamble():
    return (Node("import numpy as np") +
            Node("import matplotlib.pyplot as plt") +
            Node("from glue.core import Data"))


def export(obj):
    if hasattr(obj, '__script__'):
        return obj.__script__()

    for typ in type(obj).mro():
        if typ in _export_table:
            return _export_table[typ](obj)

    raise TypeError("Don't know how to export %s to script" % obj)


def show():
    return Node("plt.show()")


def spacer(n=1):
    return Node("\n" * n)

if __name__ == "__main__":
    from .scatter_client import ScatterClient

    import numpy as np
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    d = Data(x=x, y=y, label='data')
    dc = DataCollection(d)

    s = ScatterClient(dc)
    s.add_data(d)
    s.xatt = d.id['x']
    s.yatt = d.id['y']

    print generate_code(preamble() +
                        spacer(2) +
                        export(dc) +
                        spacer(2) +
                        export(s) +
                        show())
