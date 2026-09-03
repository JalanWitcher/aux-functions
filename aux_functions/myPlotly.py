import generalAux

def PreparaFiguraPlots(nPlots, sub_title=None, cols=2, type='xy', **kwrd):
    from plotly.subplots import make_subplots
    if nPlots < cols:
        cols = nPlots

    lastRowFull = nPlots % cols == 0

    rows = (nPlots) // cols + 1*(not lastRowFull)
    specs = []

    for row in range(rows - 1):
        specs.append(cols*[{"type": type}])

    if lastRowFull:
        specs.append(cols*[{"type": type}])
        jump = 0

    else:
        lastCols = nPlots % cols
        if  generalAux.isEven(cols) == generalAux.isEven(lastCols):
            jump = (cols - lastCols) // 2
            specs.append(jump*[None] + lastCols*[{"type": type, "colspan":1}] + jump*[None])
        else:
            padding = 2.5 * 0.2 / cols
            jump = (cols - lastCols - 1) // 2
            specs.append(jump*[None] + lastCols*[{"type": type, "colspan":2, "l": padding, "r": padding}] + (jump+1)*[None])

    fig = make_subplots(rows=rows, cols=cols, specs=specs, horizontal_spacing=0.025, vertical_spacing=0.25/rows,
                        subplot_titles=[f'Plot {i+1:d}' + (f'- {sub_title[i]}' if sub_title[i] else ' ') if sub_title else ' ' for i in range(nPlots) ], **kwrd)

    return fig

def updateLayoutSub(fig, showlegend=False, **kwrd):
    rows, cols = fig._get_subplot_rows_columns()
    rows, cols = rows[-1], cols[-1]
    height = kwrd.pop('height', 600*rows)
    width = kwrd.pop('width', 700*cols)
    fig.update_layout(height=height, width=width, showlegend=showlegend, **kwrd)
    if cols == 1 and showlegend:
        for i in range(1, rows + 1):
            yaxis = fig.layout.yaxis if i == 1 else getattr(fig.layout, f'yaxis{i}')

            y0, y1 = yaxis.domain

            fig.update_layout({f'legend{i}':dict(x=1.02, y=(y0 + y1) / 2, xanchor='left', yanchor='middle', orientation='v', groupclick='toggleitem')})