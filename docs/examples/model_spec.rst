Specifying utilities
====================

microPURC is linear in parameters: the utility of link :math:`\ell` is
:math:`z_\ell^\top \beta`, where :math:`z_\ell` collects the link's attributes.
A model is a set of terms, each mapping one regressor to one parameter
:math:`\beta`. The :mod:`micropurc.spec` package lets you declare these terms
either with a Python DSL or with the serializable schema they lower to.

The DSL
-------

Each summand is ``beta(name) * <attribute expression>``, optionally gated by a
covariate segment and optionally scaled by a constant. Attribute expressions
may be a single attribute, a product of attributes (an interaction), or an
attribute under a scalar transform (``log``, ``power``, ``standardize``).

.. code-block:: python

    from micropurc import beta, attr, seg, log, power

    utility = (
        beta("b_time") * attr("time")
        + beta("b_toll_low") * attr("toll") * seg("income", "<=", 3.0)
        + beta("b_toll_high") * attr("toll") * seg("income", ">", 3.0)
        + beta("b_logdist") * log(attr("length"))
        + beta("b_toll_sq") * power(attr("toll"), 2.0)
    )
    spec = utility.to_spec()

Here ``seg("income", "<=", 3.0)`` restricts the toll coefficient to trips whose
``income`` covariate is at most 3, so the toll sensitivity differs across
income segments while sharing the network's toll attribute.

The schema
----------

The DSL lowers to a :class:`~micropurc.spec.schema.ModelSpec`, a plain
dataclass that round-trips to and from JSON. This makes specifications
serializable and comparable across runs.

.. code-block:: python

    from micropurc.spec import ModelSpec

    text = spec.to_json()
    restored = ModelSpec.from_json(text)
    assert restored == spec

Compiling to a design
---------------------

:func:`~micropurc.spec.compile.compile_design` binds the named attributes to a
network's columns and produces the estimation design. Without segmentation the
design is a single matrix ``Z``; with segmentation (or attributes that vary
across variants) it is a stack of per-variant designs, or a lazy design that
the native engine evaluates without materializing ``(N, L, K)``.

.. code-block:: python

    import numpy as np
    from micropurc import Network, compile_design

    net = Network.grid(rows=4, cols=4, K=3)
    net.attribute_names = ["length", "time", "toll"]

    # Two income variants, one covariate column.
    variant_covariates = np.array([[1.0], [5.0]])
    trip_variant = np.array([0, 1, 0, 1])

    design = compile_design(
        spec,
        net,
        variant_covariates=variant_covariates,
        covariate_names=["income"],
        trip_variant=trip_variant,
        materialize=False,  # build a lazy design for the native engine
    )
