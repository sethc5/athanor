"""
athanor.viz.visualizer — interactive and static visualizations of the
concept graph.

Two renderers:
  - pyvis  → interactive HTML (best for exploration in notebook)
  - plotly → static/interactive PNG/HTML (good for embedding in reports)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from athanor.graph.models import ConceptGraph

log = logging.getLogger(__name__)


def _default_output_dir() -> Path:
    """Workspace-aware default output directory for visualizations."""
    from athanor.pipeline import workspace_root
    return workspace_root() / "outputs" / "graphs"


class GraphVisualizer:
    """Render a ConceptGraph as interactive HTML or a plotly figure."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._out = output_dir or _default_output_dir()

    # ── pyvis (interactive HTML) ─────────────────────────────────────────────

    def pyvis_html(
        self,
        graph: ConceptGraph,
        filename: str = "concept_graph.html",
        height: str = "750px",
        highlight_sparse: bool = True,
    ) -> Path:
        """Write an interactive pyvis graph to *outputs/graphs/<filename>*.

        Returns the path to the HTML file.
        """
        from pyvis.network import Network

        G = graph.to_networkx()
        sparse_edges = {
            (e.source, e.target)
            for e in graph.sparse_connections(top_k=10)
        }

        net = Network(height=height, width="100%", notebook=True, cdn_resources="remote")
        net.force_atlas_2based(gravity=-50, spring_length=100)

        # ── nodes ────────────────────────────────────────────────────────────
        max_centrality = max(
            (c.centrality for c in graph.concepts), default=1.0
        ) or 1.0

        for node, data in G.nodes(data=True):
            centrality = data.get("centrality", 0.0)
            size = 10 + 40 * (centrality / max_centrality)
            title = (
                f"<b>{node}</b><br>"
                f"{data.get('description', '')}<br>"
                f"Centrality: {centrality:.3f}<br>"
                f"Papers: {', '.join(data.get('source_papers', []))}"
            )
            net.add_node(
                node,
                label=node,
                title=title,
                size=size,
                color=self._node_color(centrality, max_centrality),
            )

        # ── edges ────────────────────────────────────────────────────────────
        for u, v, data in G.edges(data=True):
            is_sparse = (u, v) in sparse_edges or (v, u) in sparse_edges
            net.add_edge(
                u,
                v,
                title=data.get("relation", ""),
                width=1 + 3 * data.get("weight", 0.5),
                color="#e74c3c" if is_sparse else "#7f8c8d",
                dashes=is_sparse,
            )

        out_path = self._out / filename
        net.save_graph(str(out_path))
        log.info("Saved pyvis graph to %s", out_path)
        return out_path

    # ── plotly (static / embeddable) ─────────────────────────────────────────

    def plotly_figure(self, graph: ConceptGraph):
        """Return a plotly Figure of the concept graph.

        The figure is returned (not saved) so callers can show() or write_html().
        """
        import networkx as nx
        import plotly.graph_objects as go

        G = graph.to_networkx()
        pos = nx.spring_layout(G, weight="weight", seed=42)

        edge_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=0.5 + 2 * data.get("weight", 0.5), color="#aaa"),
                    hoverinfo="none",
                )
            )

        node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_hover.append(
                f"<b>{node}</b><br>{data.get('description','')}<br>"
                f"Centrality: {data.get('centrality', 0):.3f}"
            )
            node_size.append(8 + 30 * data.get("centrality", 0.1))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Centrality"),
            ),
        )

        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=f"Concept Graph — {graph.domain} ({graph.query})",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
            ),
        )
        return fig

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _node_color(centrality: float, max_centrality: float) -> str:
        """Map centrality to a blue→red gradient hex color."""
        ratio = centrality / max_centrality if max_centrality else 0
        r = int(50 + 180 * ratio)
        b = int(230 - 180 * ratio)
        return f"#{r:02x}6e{b:02x}"
