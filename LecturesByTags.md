---
layout: page
title: Topics-as-Tables
desc: "2026 Fall  UVa CS Machine Learning Lectures Organized by Tags"
---
<p><a name="topPage"></a></p>


Click on a tag to see relevant list of lectures.

<ul class="tags">
{% assign tags =  site.contents | map: 'tags' | uniq | sort %}
{% for tag in tags %}
  {% assign t = tag  %}
  <li><a href="{{ site.baseurl }}/LecturesByTags/#{{t | replace:" ","-" }}">{{ t }}</a></li>
{% endfor %}
</ul>

---

{% assign tags =  site.contents | map: 'tags' | uniq | sort %}
{% for tag in tags %}
  {% assign t = tag %}

<h1><a name="{{t | replace:" ","-" }}"></a><a class="internal" href="{{ site.baseurl }}/LecturesByTags/#{{t | replace:" ","-" }}">{{ t  }}</a></h1>

<!--- for each tag, get a table of index -->

<div class="table-scroll">
<table id="datatab3" summary="Table of Lectures" border="1">
<tr>
 <h3><b>
  <th>Title</th>
  <th>Lecture</th>
  <th class="resources-col">Resources</th>
  </b>
  </h3>
</tr>


  {% assign counter = 1 %}
  {% for post in site.contents %}
    {% if post.tags contains tag %}

  {% assign LidxStr = '' %}
  {% unless post.title contains "Section" or post.extra or post.background or post.platform %}
    {% assign Ltmp = 0 %}
    {% assign Lidx = 0 %}
    {% for p in site.contents %}
      {% unless p.title contains "Section" or p.extra or p.background or p.platform %}
        {% assign Ltmp = Ltmp | plus: 1 %}
        {% if p.url == post.url %}
          {% assign Lidx = Ltmp %}
        {% endif %}
      {% endunless %}
    {% endfor %}
    {% if Lidx < 10 %}
      {% assign LidxStr = 'L0' | append: Lidx %}
    {% else %}
      {% assign LidxStr = 'L' | append: Lidx %}
    {% endif %}
  {% endunless %}

  <tr>

  <td><a href="{{ site.baseurl }}{{ post.url }}">{% if LidxStr != '' %}{{ LidxStr }}: {% endif %}{{ post.title }} </a></td>

  {% if post.lecture %}
  <td><a href="{{ site.baseurl }}/Lectures/{{ post.lecture }}.pdf"  target="_blank">Slide</a></td>
  {% else %}
  <td></td>
  {% endif %}

  {% include resource-stats.html post=post %}

  <td class="resources-col">
  <a href="{{ site.baseurl }}{{ post.url }}">
  {% if vcount > 0 %}{{ vcount }} video{% if vcount != 1 %}s{% endif %}{% endif %}
  {% if vcount > 0 and rcount > 0 %} &middot; {% endif %}
  {% if rcount > 0 %}{{ rcount }} reading{% if rcount != 1 %}s{% endif %}{% endif %}
  {% if rcount > 0 and ccount > 0 %} &middot; {% endif %}
  {% if ccount > 0 %}{{ ccount }} coding-read{% if ccount != 1 %}s{% endif %}{% endif %}
  {% if vcount == 0 and rcount == 0 and ccount == 0 %}&mdash;{% endif %}
  </a>
  </td>

  </tr>

  {% assign counter=counter | plus:1 %}
  {% endif %}
{% endfor %}
</table>
</div>

<!--- for each tag, present its posts in orders -->


{% endfor %}


<div class="back-to-top-box">
<a class="back-to-top-link" href="#topPage" title="Back to Top">BackTop</a>
</div>
