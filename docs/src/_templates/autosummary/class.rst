{% import "_macros.rst.jinja" as macros %}
{% include "_header.rst.jinja" %}

{% set opts = get_classdoc_opts(fullname) %}
{% set warning = get_api_warning(fullname) %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {{ macros.own_summaries_and_autoobjs(
         name,
         opts,
         warning,
         methods,
         all_methods,
         attributes,
         all_attributes,
         inherited_members) }}