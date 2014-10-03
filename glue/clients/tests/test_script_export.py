from ..script_export import Node, generate_code, export, unindent
from ...core import Data, DataCollection
from ..scatter_client import ScatterClient
from ..histogram_client import HistogramClient


def test_one_line():
    c = Node("123")

    assert generate_code([c]) == "123"


def test_two_line():
    c = [Node("1"),
         Node("2")]
    assert generate_code(c) == "1\n2"


def test_priority_sorted():
    c = [Node("1", priority=1),
         Node("2", priority=0)]
    assert generate_code(c) == "2\n1"


def test_dependency_sorted():
    a = 1
    c = [Node("1", needs=[a]),
         Node("2", gives=a)]
    assert generate_code(c) == "2\n1"


def test_dependency_then_priority_sorted():
    a = 1
    c = [Node("1", needs=[a], priority=10),
         Node("2", needs=[a], priority=5),
         Node("3", gives=a)]
    assert generate_code(c) == "3\n2\n1"


def test_multi_gives():
    a = 1
    b = 2
    c = [Node("1", needs=[a], priority=5),
         Node("2", needs=[b], priority=15),
         Node("3", gives=[a, b])]

    assert generate_code(c) == "3\n1\n2"


def test_token_expansion():
    a = [Node("{x}", tokens=dict(x=Data(label='test')))]
    assert generate_code(a) == "test"


def test_token_collision():
    a = [Node("{x}", tokens=dict(x=Data(label='test'))),
         Node("{x}", tokens=dict(x=Data(label="test2")))]

    assert generate_code(a) == "test\ntest2"


def test_code_block():
    x = 5
    a = Node("1") + Node("2", gives=x)
    b = Node("3", needs=x)

    assert generate_code([a, b]) == "1\n2\n3"


def test_consistent_node_names():
    d = Data(label='test')
    a = [Node("{x}+1", tokens=dict(x=d)),
         Node("{x}+2", tokens=dict(x=d))]
    assert generate_code(a) == "test+1\ntest+2"


def test_resolve_label_collisions():
    d1 = Data(label='test')
    d2 = Data(label='test')

    a = [Node("{x}+1", tokens=dict(x=d1)),
         Node("{x}+2", tokens=dict(x=d2))]
    assert generate_code(a) in ("test+1\ntest_0+2", "test_0+1\ntest+2")


def test_save_scatter():
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='d1')
    dc = DataCollection(d)
    s = ScatterClient(dc)
    s.add_data(d)

    s.xatt = d.id['x']
    s.yatt = d.id['y']
    s.xmin = 0
    s.xmax = 4
    s.ymin = 1
    s.ymax = 5
    s.ylog = True
    s.xflip = True

    result = export(s)

    expected = """
    plt.figure()
    plt.plot({l0}[{x}].ravel(), {l0}[{y}].ravel(),
             markersize=3, mec='none',
             mew=3, mfc='#373737', linestyle='None',
             alpha=0.5, zorder=1)

    plt.set_xlabel(u'x')
    plt.set_ylabel(u'y')
    plt.set_xlim((4.0, 0.0))
    plt.set_ylim((1.0, 5.0))
    plt.set_xscale(u'linear')
    plt.set_yscale(u'log')
    """
    expected = unindent(expected)

    assert result.code == expected
    assert result.tokens == {'x': d.id['x'],
                             'y': d.id['y'],
                             'l0': d}


def test_save_scatter_subset():
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='d1')
    dc = DataCollection(d)
    dc.new_subset_group(subset_state=d.id['x'] > 1)

    s = ScatterClient(dc)
    s.add_data(d)

    s.xatt = d.id['x']
    s.yatt = d.id['y']
    s.xmin = 0
    s.xmax = 4
    s.ymin = 1.0
    s.ymax = 5
    s.ylog = True
    s.xflip = True

    result = export(s)

    expected = """
    plt.figure()
    plt.plot({l0}[{x}].ravel(), {l0}[{y}].ravel(),
             markersize=3, mec='none',
             mew=3, mfc='#373737', linestyle='None',
             alpha=0.5, zorder=1)

    plt.plot({l1}[{x}].ravel(), {l1}[{y}].ravel(),
             markersize=7.5, mec='none',
             mew=3, mfc='#E31A1C', linestyle='None',
             alpha=0.5, zorder=2)

    plt.set_xlabel(u'x')
    plt.set_ylabel(u'y')
    plt.set_xlim((4.0, 0.0))
    plt.set_ylim((1.0, 5.0))
    plt.set_xscale(u'linear')
    plt.set_yscale(u'log')
    """

    expected = unindent(expected)
    assert result.code == expected
    assert result.tokens == {'x': d.id['x'],
                             'y': d.id['y'],
                             'l0': d,
                             'l1': d.subsets[0]}


def test_save_histogram():

    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='d1')
    dc = DataCollection(d)

    cl = HistogramClient(dc)
    cl.add_data(d)
    cl.set_component(d.id['x'])

    cl.normed = False
    cl.cumulative = False
    cl.nbins = 25
    cl.xlog = False
    cl.ylog = True
    cl.xlimits = (0, 10)
    result = export(cl)

    print result
    expected = """
    plt.figure()
    plt.hist({l0}[{att}].ravel(), bins=25, range=(0, 10),
             normed=False, cumulative=False,
             facecolor='#373737', alpha=0.5, zorder=1)
    """

    expected = unindent(expected)
    assert result.code.startswith(expected)
