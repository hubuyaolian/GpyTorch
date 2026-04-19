import json
import os
import uuid
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Configure Chinese font
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_cell_id():
    return str(uuid.uuid4())[:8]


def parse_md_to_cells(md_content):
    cells = []
    lines = md_content.split('\n')
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith('```python'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_lines,
                "id": generate_cell_id()
            })
        elif lines[i].strip().startswith('```'):
            text_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                text_lines.append(lines[i])
                i += 1
            i += 1
            if text_lines:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": text_lines,
                    "id": generate_cell_id()
                })
        elif lines[i].strip().startswith('## ') or lines[i].strip().startswith('# '):
            text_lines = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```') and not lines[i].strip().startswith('## ') and not lines[i].strip().startswith('# '):
                text_lines.append(lines[i])
                i += 1
            if any(l.strip() for l in text_lines):
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": text_lines,
                    "id": generate_cell_id()
                })
        else:
            text_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```') and not lines[i].strip().startswith('## ') and not lines[i].strip().startswith('# '):
                text_lines.append(lines[i])
                i += 1
            if any(l.strip() for l in text_lines):
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": text_lines,
                    "id": generate_cell_id()
                })

    return cells


def render_diagram(code):
    """Execute diagram code and return base64 PNG"""
    plt.close('all')
    fig = None

    # Capture the figure by mocking plt.show()
    figures = []
    original_show = plt.show

    def mock_show(*args, **kwargs):
        figs = [manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
        figures.extend(figs)

    plt.show = mock_show

    try:
        exec(code, {'plt': plt, 'np': np, 'matplotlib': matplotlib})
    except Exception as e:
        print(f"  Diagram error: {e}")
        plt.show = original_show
        return None

    plt.show = original_show

    if not figures:
        # Try to get current figure
        fig = plt.gcf()
        if fig.get_axes():
            figures = [fig]

    if not figures:
        return None

    # Save first figure as base64
    buf = io.BytesIO()
    fig = figures[0]
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')

    return img_base64


def make_diagram_md_cell(code, title=""):
    """Create a markdown cell with embedded base64 image"""
    img_base64 = render_diagram(code)
    if img_base64:
        md_content = [
            f'<p align="center"><img src="data:image/png;base64,{img_base64}" alt="{title}" width="800"></p>',
            '',
            f'*{title}*'
        ]
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_content,
            "id": generate_cell_id()
        }
    return None


def add_visual_diagrams(cells, chapter_num):
    visual_cells = []

    for idx, cell in enumerate(cells):
        visual_cells.append(cell)
        source_text = ''.join(cell.get('source', []))

        if chapter_num == 1:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n_people, title in zip(axes, [1, 5, 20],
    ['1人干活：指令清晰', '5人干活：分工明确', '20人干活：信息失真']):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    positions = np.linspace(1, 9, n_people)
    for j, pos in enumerate(positions):
        ax.plot(5, pos, 'o', color='#3498db', markersize=15)
        ax.text(5.5, pos, f'层{j+1}', fontsize=10, va='center')
    if n_people > 1:
        for j in range(len(positions)-1):
            ax.annotate('', xy=(5, positions[j+1]-0.3),
                       xytext=(5, positions[j]+0.3),
                       arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, alpha=0.7))
    if n_people == 20:
        ax.text(8, 5, '信息\\n失真!', fontsize=14, color='#e74c3c',
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
plt.suptitle('生活隐喻：为什么"人多力量大"有时会失效？', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "信息传递失真示意图")
                if dc: visual_cells.append(dc)

            if '## 5. 几何直觉' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
messages = ['"今天天气真好"', '"今天天气不错"', '"今天天气还行"',
            '"今天天气一般"', '"今天...什么来着？"']
colors = ['#2ecc71', '#82e0aa', '#f9e79f', '#f5b041', '#e74c3c']
alphas = [1.0, 0.85, 0.65, 0.45, 0.25]
for i, (msg, color, alpha) in enumerate(zip(messages, colors, alphas)):
    y = 4 - i * 0.8
    ax.barh(y, 1.0 - i*0.2, left=i*0.2, color=color, alpha=alpha, height=0.5)
    ax.text(0.5 + i*0.2, y, f'第{i+1}层: {msg}', fontsize=11,
           va='center', fontweight='bold' if i == 0 else 'normal')
    if i < len(messages) - 1:
        ax.annotate('', xy=(0.5 + (i+1)*0.2, y - 0.35),
                   xytext=(0.5 + i*0.2, y - 0.15),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.set_xlim(-0.3, 6)
ax.set_ylim(0, 5)
ax.set_title('传话游戏：信息在层间传递中逐渐丢失', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()
""", "传话游戏信息衰减示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 2:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n, title in zip(axes, [4, 20, 100],
    ['4人接力：顺利', '20人接力：频繁掉棒', '100人接力：几乎不可能']):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    n_show = min(n, 15)
    positions = np.linspace(1, 9, n_show)
    for j, pos in enumerate(positions):
        color = '#2ecc71' if j < n_show//3 else ('#f9e79f' if j < 2*n_show//3 else '#e74c3c')
        ax.plot(5, pos, 's', color=color, markersize=10)
        if j < n_show - 1:
            ax.annotate('', xy=(5, positions[j+1]+0.2), xytext=(5, pos-0.2),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.text(8, 5, f'交接{n-1}次\\n掉棒概率\\n{(n-1)*0.05:.0%}',
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0' if n > 10 else '#e0ffe0'))
plt.suptitle('接力赛隐喻：交接次数越多，掉棒概率越高', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "接力赛掉棒示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 3:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('普通网络：从零画图', fontsize=14, fontweight='bold')
ax.axis('off')
ax.annotate('输入 x', xy=(2, 7), fontsize=13, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3'))
ax.annotate('', xy=(2, 5.5), xytext=(2, 6.5), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('Layer1\\n重画整个图', xy=(2, 5), fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f'))
ax.annotate('', xy=(2, 3.5), xytext=(2, 4.3), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('Layer2\\n再重画整个图', xy=(2, 3), fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f'))
ax.annotate('', xy=(2, 1.5), xytext=(2, 2.3), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('输出 H(x)', xy=(2, 1), fontsize=13, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8'))
ax.text(6, 4, '每一层必须\\n从零学起\\n学不好就全崩', fontsize=12, ha='center', va='center',
       color='#e74c3c', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe0e0', alpha=0.8))
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('残差网络：标注修改', fontsize=14, fontweight='bold')
ax.axis('off')
ax.annotate('输入 x', xy=(2, 7), fontsize=13, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3'))
ax.annotate('', xy=(2, 5.5), xytext=(2, 6.5), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('Layer1\\n标注修改 F1(x)', xy=(2, 5), fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4efdf'))
ax.annotate('', xy=(2, 3.5), xytext=(2, 4.3), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('Layer2\\n标注修改 F2(x)', xy=(2, 3), fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4efdf'))
ax.annotate('', xy=(2, 1.5), xytext=(2, 2.3), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('输出 F(x)+x', xy=(2, 1), fontsize=13, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3'))
ax.annotate('', xy=(6.5, 1.2), xytext=(6.5, 6.8),
           arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3, connectionstyle='arc3,rad=-0.3'))
ax.text(8, 4, '近道!\\nx 直接传过去\\nF(x)=0 也行', fontsize=12, ha='center', va='center',
       color='#27ae60', fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0ffe0', alpha=0.8))
plt.suptitle('核心对比：从零画图 vs 标注修改', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "普通网络 vs 残差网络 信息流对比图")
                if dc: visual_cells.append(dc)

            if '## 5. 信息流的几何直觉' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.set_title('普通网络信息流', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
layers_x = [1, 3, 5, 7, 9, 11]
for i, x in enumerate(layers_x):
    ax.add_patch(plt.Rectangle((x-0.4, 2), 0.8, 2, facecolor='#fadbd8', edgecolor='#e74c3c', lw=2))
    ax.text(x, 3, f'L{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    if i < len(layers_x) - 1:
        ax.annotate('', xy=(layers_x[i+1]-0.5, 3), xytext=(x+0.5, 3),
                   arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
ax.text(6, 0.8, '必须经过每一层，任何一层出问题都会影响结果', fontsize=11, ha='center', color='#e74c3c', style='italic')
ax = axes[1]
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.set_title('残差网络信息流', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
for i, x in enumerate(layers_x):
    ax.add_patch(plt.Rectangle((x-0.4, 2), 0.8, 2, facecolor='#d5f5e3', edgecolor='#27ae60', lw=2))
    ax.text(x, 3, f'L{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    if i < len(layers_x) - 1:
        ax.annotate('', xy=(layers_x[i+1]-0.5, 3), xytext=(x+0.5, 3),
                   arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax.annotate('', xy=(11-0.5, 5), xytext=(1+0.5, 5),
           arrowprops=dict(arrowstyle='->', color='#2980b9', lw=3, connectionstyle='arc3,rad=-0.15'))
ax.text(6, 5.3, '近道 (skip connection): x 直接传过去', fontsize=11, ha='center', color='#2980b9', fontweight='bold')
ax.text(6, 0.8, '信息有两条路：走远路(变换) 或 走近道(直通)', fontsize=11, ha='center', color='#27ae60', style='italic')
plt.suptitle('信息流对比：普通网络 vs 残差网络', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "信息流路径对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 4:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Photoshop 图层叠加隐喻', fontsize=16, fontweight='bold')
ax.add_patch(plt.Rectangle((1, 1), 4, 5, facecolor='#d5f5e3', edgecolor='#27ae60', lw=3))
ax.text(3, 3.5, '底层图层\\n(原始照片 x)', fontsize=13, ha='center', va='center', fontweight='bold', color='#27ae60')
ax.add_patch(plt.Rectangle((4.5, 3), 4, 3, facecolor='#fdebd0', edgecolor='#e67e22', lw=3, alpha=0.8))
ax.text(6.5, 4.5, '调整图层\\n(修正量 F(x))\\n亮度+10, 对比度+5', fontsize=12, ha='center', va='center', color='#e67e22')
ax.annotate('', xy=(8.5, 4), xytext=(8, 4), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))
ax.add_patch(plt.Rectangle((9, 1), 2.5, 5, facecolor='#d4e6f1', edgecolor='#2980b9', lw=3))
ax.text(10.25, 3.5, '最终效果\\nF(x)+x', fontsize=13, ha='center', va='center', fontweight='bold', color='#2980b9')
ax.text(6, 0.3, '关键：调整图层="无调整"(F(x)=0) -> 最终效果=原始照片 -> 不会变差！',
       fontsize=12, ha='center', color='#e74c3c', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
plt.tight_layout()
plt.show()
""", "图层叠加隐喻示意图")
                if dc: visual_cells.append(dc)

            if '## 6. 为什么 ReLU 放在加法之后' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
ax.set_title('错误做法：ReLU 在加法之前', fontsize=13, color='#e74c3c', fontweight='bold')
ax.axis('off')
boxes = [('x', 1, 2, '#d5f5e3'), ('Layer1', 3, 2, '#fdebd0'), ('Layer2', 5, 2, '#fdebd0'), ('ReLU', 7, 2, '#fadbd8'), ('+x', 9, 2, '#d4e6f1')]
for name, bx, by, color in boxes:
    ax.add_patch(plt.Rectangle((bx-0.4, by-0.4), 0.8, 0.8, facecolor=color, edgecolor='gray', lw=1.5))
    ax.text(bx, by, name, ha='center', va='center', fontsize=9, fontweight='bold')
    if bx < 9:
        next_bx = boxes[boxes.index((name, bx, by, color)) + 1][1]
        ax.annotate('', xy=(next_bx-0.5, by), xytext=(bx+0.5, by), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(5, 0.5, 'ReLU 截断了 F(x) 的负值 -> 残差信号不完整', fontsize=10, ha='center', color='#e74c3c')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
ax.set_title('正确做法：ReLU 在加法之后', fontsize=13, color='#27ae60', fontweight='bold')
ax.axis('off')
boxes = [('x', 1, 2, '#d5f5e3'), ('Layer1', 3, 2, '#fdebd0'), ('Layer2', 5, 2, '#fdebd0'), ('+x', 7, 2, '#d4e6f1'), ('ReLU', 9, 2, '#fadbd8')]
for name, bx, by, color in boxes:
    ax.add_patch(plt.Rectangle((bx-0.4, by-0.4), 0.8, 0.8, facecolor=color, edgecolor='gray', lw=1.5))
    ax.text(bx, by, name, ha='center', va='center', fontsize=9, fontweight='bold')
    if bx < 9:
        next_bx = boxes[boxes.index((name, bx, by, color)) + 1][1]
        ax.annotate('', xy=(next_bx-0.5, by), xytext=(bx+0.5, by), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(5, 0.5, '先叠加完整残差信号，再统一激活 -> 信息完整', fontsize=10, ha='center', color='#27ae60')
plt.suptitle('ReLU 位置的关键区别', fontsize=15, y=1.05)
plt.tight_layout()
plt.show()
""", "ReLU 位置对比示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 5:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title('Plain Net：无安全绳攀岩', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
heights = np.linspace(1, 9, 8)
for i, h in enumerate(heights):
    ax.plot(5, h, 's', color='#e74c3c' if i < 4 else '#fadbd8', markersize=12)
    ax.text(5.5, h, f'层{i+1}', fontsize=9, va='center')
    if i < len(heights) - 1:
        ax.annotate('', xy=(5, heights[i+1]+0.2), xytext=(5, h-0.2), arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
ax.text(2, 5, '每一步都必须踩稳\\n踩空就一路滑到底', fontsize=11, ha='center', color='#e74c3c', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title('ResNet：有安全绳攀岩', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
for i, h in enumerate(heights):
    ax.plot(5, h, 's', color='#27ae60', markersize=12)
    ax.text(5.5, h, f'层{i+1}', fontsize=9, va='center')
    if i < len(heights) - 1:
        ax.annotate('', xy=(5, heights[i+1]+0.2), xytext=(5, h-0.2), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
for i in range(0, len(heights)-1, 2):
    ax.annotate('', xy=(7.5, heights[i+1]), xytext=(7.5, heights[i]),
               arrowprops=dict(arrowstyle='->', color='#2980b9', lw=3, connectionstyle='arc3,rad=-0.3'))
    ax.text(8.5, (heights[i]+heights[i+1])/2, '安全绳', fontsize=9, ha='center', color='#2980b9', fontweight='bold', rotation=90)
ax.text(2, 5, '踩空了也不怕\\n安全绳拉住你', fontsize=11, ha='center', color='#27ae60', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
plt.suptitle('攀岩隐喻：有安全绳 vs 无安全绳', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "安全绳攀岩隐喻示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 6:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('乡间小路：梯度越走越慢', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
villages = np.linspace(1, 9, 6)
speeds = [1.0, 0.7, 0.45, 0.25, 0.12, 0.05]
for i, (v, s) in enumerate(zip(villages, speeds)):
    size = max(5, s * 20)
    ax.plot(v, 4, 'o', color=plt.cm.Reds(1-s), markersize=size+5)
    ax.text(v, 2.5, f'x{s:.2f}', ha='center', fontsize=9, color='#e74c3c')
    if i < len(villages) - 1:
        ax.annotate('', xy=(villages[i+1]-0.3, 4), xytext=(v+0.3, 4), arrowprops=dict(arrowstyle='->', color='gray', lw=max(0.5, s*3)))
ax.text(5, 6.5, '每经过一个村庄，速度减半\\n最终几乎走不动', fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('高速公路：梯度畅通无阻', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
for i, v in enumerate(villages):
    ax.plot(v, 4, 'o', color='#27ae60', markersize=15)
    ax.text(v, 2.5, f'x1.0', ha='center', fontsize=9, color='#27ae60', fontweight='bold')
    if i < len(villages) - 1:
        ax.annotate('', xy=(villages[i+1]-0.3, 4), xytext=(v+0.3, 4), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
ax.annotate('', xy=(9, 6.5), xytext=(1, 6.5), arrowprops=dict(arrowstyle='->', color='#2980b9', lw=4))
ax.text(5, 7, '高速直达！加法的 +1 保证梯度不消失', fontsize=12, ha='center', color='#2980b9', fontweight='bold')
plt.suptitle('梯度传播隐喻：乡间小路 vs 高速公路', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "梯度传播隐喻示意图")
                if dc: visual_cells.append(dc)

            if '## 3. 数学直觉' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
depths = np.arange(1, 51)
grad_plain = 0.9 ** depths
ax.plot(depths, grad_plain, color='#e74c3c', linewidth=3, label='普通网络: 0.9^n')
ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='梯度消失阈值')
ax.fill_between(depths, 0, grad_plain, alpha=0.15, color='#e74c3c')
ax.set_xlabel('网络深度 (层数)', fontsize=12)
ax.set_ylabel('梯度大小', fontsize=12)
ax.set_title('普通网络：梯度指数衰减', fontsize=14, color='#e74c3c')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_yscale('log')
ax = axes[1]
grad_res = np.ones_like(depths, dtype=float) * 1.0
ax.plot(depths, grad_res, color='#27ae60', linewidth=3, label='残差网络: >=1.0')
ax.axhline(y=1.0, color='#2980b9', linestyle='--', alpha=0.5, label='保底线 (+1)')
ax.fill_between(depths, 1.0, grad_res, alpha=0.15, color='#27ae60')
ax.set_xlabel('网络深度 (层数)', fontsize=12)
ax.set_ylabel('梯度大小', fontsize=12)
ax.set_title('残差网络：梯度始终>=1', fontsize=14, color='#27ae60')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.suptitle('梯度传播的核心区别：乘法衰减 vs 加法保底', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "梯度衰减 vs 梯度保底 对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 7:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('方式一：从头重写', fontsize=13, fontweight='bold', color='#e74c3c')
ax.axis('off')
ax.text(5, 6, '重写整份文档\\n(必须一模一样)', fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fadbd8'))
ax.text(5, 3, '极难！稍有偏差\\n就不等于原文', fontsize=11, ha='center', color='#e74c3c')
ax.text(5, 1, 'H(x) = x 很难学', fontsize=12, ha='center', fontweight='bold', color='#e74c3c', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('方式二：直接提交原稿', fontsize=13, fontweight='bold', color='#f39c12')
ax.axis('off')
ax.text(5, 6, '提交原稿\\n(理想但普通网络做不到)', fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdebd0'))
ax.text(5, 3, '普通网络没有\\n"不做事"的选项', fontsize=11, ha='center', color='#f39c12')
ax.text(5, 1, '普通网络无法选择躺平', fontsize=12, ha='center', fontweight='bold', color='#f39c12', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0'))
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('方式三：标注"无修改"', fontsize=13, fontweight='bold', color='#27ae60')
ax.axis('off')
ax.text(5, 6, '标注"无修改"\\nF(x) = 0', fontsize=12, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3'))
ax.text(5, 3, '太简单了！\\n权重接近0就行', fontsize=11, ha='center', color='#27ae60')
ax.text(5, 1, '残差网络：F(x)=0 轻松学', fontsize=12, ha='center', fontweight='bold', color='#27ae60', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
plt.suptitle('三种"不做事情"的方式：恒等映射的智慧', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "恒等映射三种方式对比图")
                if dc: visual_cells.append(dc)

            if '## 3. 几何直觉' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_title('普通网络：寻找 H(x)=x 的精确点', fontsize=13, fontweight='bold', color='#e74c3c')
ax.set_xlabel('权重 w1', fontsize=11); ax.set_ylabel('权重 w2', fontsize=11)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(2*np.cos(theta), 2*np.sin(theta), '--', color='gray', alpha=0.3)
ax.plot(0, 0, 'x', color='#e74c3c', markersize=20, markeredgewidth=3)
ax.text(0.3, 0.3, 'H(x)=x\\n(极难找到!)', fontsize=11, color='#e74c3c', fontweight='bold')
np.random.seed(42)
attempts_x = np.random.randn(15) * 1.5
attempts_y = np.random.randn(15) * 1.5
ax.scatter(attempts_x, attempts_y, c='#f5b041', s=30, alpha=0.6, zorder=3)
ax.text(-2.5, -2.5, '随机初始化\\n离目标很远', fontsize=10, color='#f5b041')
ax.grid(True, alpha=0.3)
ax = axes[1]
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_title('残差网络：F(x)=0 就在起点', fontsize=13, fontweight='bold', color='#27ae60')
ax.set_xlabel('权重 w1', fontsize=11); ax.set_ylabel('权重 w2', fontsize=11)
ax.plot(2*np.cos(theta), 2*np.sin(theta), '--', color='gray', alpha=0.3)
ax.plot(0, 0, 'o', color='#27ae60', markersize=20, markeredgewidth=3)
ax.text(0.3, 0.3, 'F(x)=0\\n(就在起点!)', fontsize=11, color='#27ae60', fontweight='bold')
ax.scatter(attempts_x * 0.3, attempts_y * 0.3, c='#82e0aa', s=30, alpha=0.6, zorder=3)
circle = plt.Circle((0, 0), 0.5, fill=False, color='#27ae60', lw=2, linestyle='--')
ax.add_patch(circle)
ax.text(-2.5, -2.5, '初始化就\\n在目标附近', fontsize=10, color='#27ae60')
ax.grid(True, alpha=0.3)
plt.suptitle('优化视角：普通网络 vs 残差网络的起点', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "恒等映射优化起点对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 8:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('MiniResNet 架构图：乐高积木式搭建', fontsize=16, fontweight='bold')
components = [
    (2, 8.5, 3, 1, '#3498db', 'Stem\\n输入->特征'),
    (6, 8.5, 3, 1, '#27ae60', 'Layer1\\n32维 x 2块'),
    (10, 8.5, 3, 1, '#e67e22', 'Layer2\\n64维 x 2块'),
    (6, 6.5, 3, 1, '#9b59b6', 'Layer3\\n128维 x 2块'),
    (10, 6.5, 3, 1, '#e74c3c', 'Head\\n特征->输出'),
]
for x, y, w, h, color, label in components:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.3, edgecolor=color, lw=2, joinstyle='round'))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
ax.annotate('', xy=(5.8, 9), xytext=(5.2, 9), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('', xy=(9.8, 9), xytext=(9.2, 9), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('', xy=(7.5, 7.7), xytext=(7.5, 8.3), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('', xy=(9.8, 7), xytext=(9.2, 7), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.text(7, 4.5, '每个 Layer 内部：', fontsize=13, fontweight='bold')
ax.text(7, 3.5, '[ResidualBlock] -> [ResidualBlock] -> ...', fontsize=12, ha='center', family='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3'))
ax.text(7, 2.5, '维度变化时：[ProjectionBlock] -> [ResidualBlock] -> ...', fontsize=12, ha='center', family='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdebd0'))
ax.text(7, 1, '同样的积木，不同的堆叠 -> 不同的 ResNet', fontsize=13, ha='center', fontweight='bold', color='#2c3e50', bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8'))
plt.tight_layout()
plt.show()
""", "MiniResNet 架构示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 9:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('BasicBlock：两人搬砖', fontsize=14, fontweight='bold', color='#3498db')
ax.axis('off')
widths = [3, 3]
labels = ['Linear\\n(256->256)', 'Linear\\n(256->256)']
colors_bb = ['#aed6f1', '#85c1e9']
for i, (w, label, color) in enumerate(zip(widths, labels, colors_bb)):
    x = 2 + i * 3.5
    ax.add_patch(plt.Rectangle((x, 2.5), 2.5, 3, facecolor=color, edgecolor='#3498db', lw=2))
    ax.text(x + 1.25, 4, label, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 1:
        ax.annotate('', xy=(x+2.7, 4), xytext=(x+2.5, 4), arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
ax.annotate('', xy=(8.5, 1.5), xytext=(2, 1.5), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3, connectionstyle='arc3,rad=-0.2'))
ax.text(5, 0.5, '+ x (跳跃连接)', fontsize=11, ha='center', color='#27ae60', fontweight='bold')
ax.text(5, 7, '参数量: 256x256x2 = 131K', fontsize=12, ha='center', fontweight='bold', color='#3498db')
ax = axes[1]
ax.set_xlim(0, 12); ax.set_ylim(0, 8)
ax.set_title('Bottleneck：压缩->加工->恢复', fontsize=14, fontweight='bold', color='#e67e22')
ax.axis('off')
widths_bn = [2, 1.5, 2]
labels_bn = ['1x1压缩\\n(256->64)', '3x3加工\\n(64->64)', '1x1恢复\\n(64->256)']
colors_bn = ['#f9e79f', '#f5cba7', '#fdebd0']
x_positions = [1, 4, 7]
for i, (x, w, label, color) in enumerate(zip(x_positions, widths_bn, labels_bn, colors_bn)):
    ax.add_patch(plt.Rectangle((x, 2.5), 2.2, 3, facecolor=color, edgecolor='#e67e22', lw=2))
    ax.text(x + 1.1, 4, label, ha='center', va='center', fontsize=9, fontweight='bold')
    if i < 2:
        ax.annotate('', xy=(x_positions[i+1]-0.1, 4), xytext=(x+2.3, 4), arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
ax.annotate('', xy=(9.5, 1.5), xytext=(1, 1.5), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3, connectionstyle='arc3,rad=-0.15'))
ax.text(5, 0.5, '+ x (跳跃连接)', fontsize=11, ha='center', color='#27ae60', fontweight='bold')
ax.text(6, 7, '参数量: 256x64+64x64+64x256 = 24K (省82%!)', fontsize=11, ha='center', fontweight='bold', color='#e67e22')
plt.suptitle('BasicBlock vs Bottleneck 结构对比', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "BasicBlock vs Bottleneck 结构图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 10:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('从零盖一栋楼：MiniResNet 建筑隐喻', fontsize=16, fontweight='bold')
steps = [
    (7, 10.5, 4, 1, '#3498db', 'Stem: 输入2维 -> 32维特征'),
    (7, 9, 4, 1, '#27ae60', 'Layer1: 32维 x 2个残差块'),
    (7, 7.5, 4, 1, '#e67e22', 'Layer2: 64维 x 2个残差块'),
    (7, 6, 4, 1, '#9b59b6', 'Layer3: 128维 x 2个残差块'),
    (7, 4.5, 4, 1, '#e74c3c', 'Head: 128维 -> 3类输出'),
]
for x, y, w, h, color, label in steps:
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, alpha=0.25, edgecolor=color, lw=2))
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
for i in range(len(steps)-1):
    y_from = steps[i][1] - steps[i][3]/2
    y_to = steps[i+1][1] + steps[i+1][3]/2
    ax.annotate('', xy=(7, y_to+0.1), xytext=(7, y_from-0.1), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.annotate('', xy=(11, 4.5), xytext=(11, 10.5), arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=3))
ax.text(12, 7.5, '电梯\\n(跳跃连接)\\n信息直达', fontsize=12, ha='center', va='center', color='#2980b9', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#d6eaf8', alpha=0.8))
ax.text(7, 2.5, '验收：检查分类准确率', fontsize=13, ha='center', fontweight='bold', color='#2c3e50', bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8'))
ax.text(7, 1.2, '关键：每层都有"电梯"(跳跃连接)，信息不会在楼层间丢失', fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
plt.tight_layout()
plt.show()
""", "盖楼隐喻示意图")
                if dc: visual_cells.append(dc)

            if '## 5. 阶段四全回顾' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('阶段四知识演进路线图', fontsize=16, fontweight='bold')
topics = [
    (5, 11, '#e74c3c', 'Day 01-02: 退化问题', '网络越深越蠢？信息传递丢失'),
    (5, 9.5, '#f39c12', 'Day 03-04: 残差连接', 'out = F(x) + x，一行代码的奇迹'),
    (5, 8, '#27ae60', 'Day 05: 实战验证', 'Plain Net vs ResNet，残差逆转退化'),
    (5, 6.5, '#2980b9', 'Day 06: 梯度高速公路', '加法的+1保证梯度不消失'),
    (5, 5, '#9b59b6', 'Day 07: 恒等映射', 'F(x)=0，最简单的"不做事"'),
    (5, 3.5, '#1abc9c', 'Day 08: 架构设计', 'Stem+Layer+Head，积木式搭建'),
    (5, 2, '#e67e22', 'Day 09: ResNet家族', 'BasicBlock vs Bottleneck'),
    (5, 0.5, '#2c3e50', 'Day 10: 从零搭建', '亲手打造 MiniResNet'),
]
for x, y, color, title, desc in topics:
    ax.add_patch(plt.Rectangle((1, y-0.4), 8, 0.8, facecolor=color, alpha=0.15, edgecolor=color, lw=2, joinstyle='round'))
    ax.text(1.3, y, title, fontsize=12, fontweight='bold', color=color, va='center')
    ax.text(5.5, y, desc, fontsize=10, color='#555555', va='center')
for i in range(len(topics)-1):
    y_from = topics[i][1] - 0.4
    y_to = topics[i+1][1] + 0.4
    ax.annotate('', xy=(5, y_to+0.05), xytext=(5, y_from-0.05), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
plt.tight_layout()
plt.show()
""", "阶段四知识路线图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 11:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('金鱼：看完就忘', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
words = ['我', '爱', '深度', '学习']
for i, w in enumerate(words):
    ax.text(2 + i*2, 6, w, fontsize=16, ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#e74c3c'))
    ax.text(2 + i*2, 4, '忘了!', fontsize=12, ha='center', color='#e74c3c', alpha=0.3+i*0.2)
ax.text(5, 2, '每个词独立处理\\n看不到前后关系', fontsize=12, ha='center', color='#e74c3c',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('RNN：边看边记', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
for i, w in enumerate(words):
    ax.text(2 + i*2, 6, w, fontsize=16, ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'))
    if i > 0:
        ax.annotate('', xy=(2 + i*2 - 0.5, 5), xytext=(2 + (i-1)*2 + 0.5, 5),
                   arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax.text(2 + i*2, 4, f'h{i+1}', fontsize=12, ha='center', color='#27ae60', fontweight='bold')
ax.text(5, 2, '逐词处理 + 隐藏状态\\n记住之前的信息', fontsize=12, ha='center', color='#27ae60',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
plt.suptitle('金鱼 vs RNN：没有记忆 vs 有记忆', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "金鱼记忆 vs RNN 记忆对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 12:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('RNN 隐藏状态：边看边记的读书人', fontsize=16, fontweight='bold')
words = ['我', '喜欢', '深度', '学习']
for i, w in enumerate(words):
    x = 1.5 + i * 2.5
    ax.add_patch(plt.Rectangle((x-0.8, 5), 1.6, 1.5, facecolor='#d5f5e3', edgecolor='#27ae60', lw=2))
    ax.text(x, 5.75, w, fontsize=14, ha='center', va='center', fontweight='bold', color='#27ae60')
    ax.add_patch(plt.Rectangle((x-0.8, 2.5), 1.6, 1.5, facecolor='#d4e6f1', edgecolor='#2980b9', lw=2))
    memos = ['主语="我"', '我喜欢某事', '我喜欢深度某事', '我喜欢深度学习!']
    ax.text(x, 3.25, f'h{i+1}\\n{memos[i]}', fontsize=9, ha='center', va='center', color='#2980b9')
    if i > 0:
        ax.annotate('', xy=(x-1, 3.25), xytext=(x-1.7, 3.25),
                   arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))
        ax.text(x-1.35, 4.3, 'h_{t-1}', fontsize=9, ha='center', color='#e67e22', fontweight='bold')
    ax.annotate('', xy=(x, 5), xytext=(x, 4), arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
ax.text(6, 1, 'h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)', fontsize=13, ha='center',
       color='#2c3e50', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f9e79f'))
plt.tight_layout()
plt.show()
""", "RNN 隐藏状态工作原理图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 13:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('RNN 梯度消失：金鱼7秒记忆', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
steps = np.arange(1, 21)
grad = 0.5 ** steps
ax.bar(steps, grad, color=[plt.cm.Reds(g) for g in grad], edgecolor='gray', alpha=0.8)
ax.set_xlabel('时间步', fontsize=12)
ax.set_ylabel('梯度大小', fontsize=12)
ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
ax.text(15, 0.05, '有效阈值', fontsize=10, color='gray')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('LSTM 梯度保持：大象记忆', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
lstm_grad = np.ones(20) * 0.8 + np.random.randn(20) * 0.05
lstm_grad = np.clip(lstm_grad, 0.6, 1.0)
ax.bar(steps, lstm_grad, color='#27ae60', edgecolor='gray', alpha=0.8)
ax.set_xlabel('时间步', fontsize=12)
ax.set_ylabel('梯度大小', fontsize=12)
ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
plt.suptitle('梯度消失 vs 梯度保持', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "RNN梯度消失 vs LSTM梯度保持对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 14:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('LSTM 门控书房：三个神奇阀门', fontsize=16, fontweight='bold')
ax.add_patch(plt.Rectangle((1, 3), 5, 5, facecolor='#fdebd0', edgecolor='#e67e22', lw=3))
ax.text(3.5, 7.5, '白板 (细胞状态 C_t)', fontsize=13, ha='center', fontweight='bold', color='#e67e22')
ax.text(3.5, 5.5, '"小明喜欢篮球\\n小红喜欢音乐"', fontsize=11, ha='center', color='#2c3e50')
gates = [
    (8, 8, '#e74c3c', '遗忘门', '擦掉旧笔记'),
    (8, 5.5, '#27ae60', '输入门', '写上新笔记'),
    (8, 3, '#3498db', '输出门', '读取信息'),
]
for x, y, color, name, desc in gates:
    ax.add_patch(plt.Rectangle((x-1, y-0.6), 2, 1.2, facecolor=color, alpha=0.2, edgecolor=color, lw=2))
    ax.text(x, y+0.2, name, fontsize=12, ha='center', fontweight='bold', color=color)
    ax.text(x, y-0.3, desc, fontsize=10, ha='center', color='#555')
    ax.annotate('', xy=(6, y), xytext=(x-1, y), arrowprops=dict(arrowstyle='<->', color=color, lw=2))
ax.text(11, 5.5, 'C_t = f_t * C_{t-1}\\n     + i_t * C_tilde\\n\\n加法更新!\\n梯度保底!', fontsize=12, ha='center',
       color='#e74c3c', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0'))
ax.text(7, 1, '核心: 加法更新 = 类似 ResNet 的 F(x)+x = 梯度无损传递', fontsize=12, ha='center',
       color='#2c3e50', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f'))
plt.tight_layout()
plt.show()
""", "LSTM 门控书房示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 15:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('金鱼 (RNN)', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
ax.text(5, 6, '7秒记忆', fontsize=20, ha='center', fontweight='bold', color='#e74c3c')
ax.text(5, 4, '只能记住最近几步\\n长距离信息全忘了', fontsize=12, ha='center', color='#555')
ax.text(5, 2, '序列>20步 ≈ 随机猜测', fontsize=12, ha='center', color='#e74c3c',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('大象 (LSTM)', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
ax.text(5, 6, '长期记忆', fontsize=20, ha='center', fontweight='bold', color='#27ae60')
ax.text(5, 4, '门控 + 加法更新\\n关键信息穿越时间', fontsize=12, ha='center', color='#555')
ax.text(5, 2, '序列100步仍高准确率', fontsize=12, ha='center', color='#27ae60',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
plt.suptitle('金鱼 vs 大象：RNN vs LSTM', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "RNN金鱼 vs LSTM大象对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 16:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('词嵌入：词语的"身份证"', fontsize=16, fontweight='bold')
words_pos = [('好', 2, 6), ('棒', 3, 6.5), ('喜欢', 4, 5.5), ('差', 8, 2), ('烂', 9, 1.5), ('讨厌', 8.5, 3)]
words_neu = [('不', 5.5, 4), ('很', 5, 4.5), ('非常', 4.5, 3.5)]
for w, x, y in words_pos:
    color = '#27ae60' if x < 6 else '#e74c3c'
    ax.plot(x, y, 'o', color=color, markersize=15, alpha=0.7)
    ax.text(x, y+0.4, w, fontsize=12, ha='center', fontweight='bold', color=color)
for w, x, y in words_neu:
    ax.plot(x, y, 's', color='#95a5a6', markersize=12, alpha=0.7)
    ax.text(x, y+0.4, w, fontsize=11, ha='center', color='#95a5a6')
ax.text(3, 7.5, '正面词', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
ax.text(8.5, 0.5, '负面词', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')
ax.text(5, 3, '中性词', fontsize=12, ha='center', color='#95a5a6')
ax.text(6, 7.5, '语义相近的词在向量空间中靠近', fontsize=12, ha='center', color='#2c3e50',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f'))
plt.tight_layout()
plt.show()
""", "词嵌入空间示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 17:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('LSTM: 三扇门的书房', fontsize=14, fontweight='bold', color='#3498db')
ax.axis('off')
gates = [('遗忘门', 2, 6, '#e74c3c'), ('输入门', 5, 6, '#27ae60'), ('输出门', 8, 6, '#3498db')]
for name, x, y, color in gates:
    ax.add_patch(plt.Rectangle((x-1, y-0.5), 2, 1, facecolor=color, alpha=0.2, edgecolor=color, lw=2))
    ax.text(x, y, name, fontsize=12, ha='center', fontweight='bold', color=color)
ax.add_patch(plt.Rectangle((2, 2), 6, 2.5, facecolor='#fdebd0', edgecolor='#e67e22', lw=2))
ax.text(5, 3.25, '白板 (细胞状态)', fontsize=13, ha='center', fontweight='bold', color='#e67e22')
ax.text(5, 1, '精细但复杂: 3门 + 2状态', fontsize=11, ha='center', color='#555')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('GRU: 两开关的笔记本', fontsize=14, fontweight='bold', color='#e67e22')
ax.axis('off')
gates2 = [('重置门', 3, 6, '#e74c3c'), ('更新门', 7, 6, '#27ae60')]
for name, x, y, color in gates2:
    ax.add_patch(plt.Rectangle((x-1.2, y-0.5), 2.4, 1, facecolor=color, alpha=0.2, edgecolor=color, lw=2))
    ax.text(x, y, name, fontsize=12, ha='center', fontweight='bold', color=color)
ax.add_patch(plt.Rectangle((2, 2), 6, 2.5, facecolor='#d5f5e3', edgecolor='#27ae60', lw=2))
ax.text(5, 3.25, '笔记本 (隐藏状态)', fontsize=13, ha='center', fontweight='bold', color='#27ae60')
ax.text(5, 1, '简洁够用: 2门 + 1状态', fontsize=11, ha='center', color='#555')
plt.suptitle('LSTM vs GRU：三扇门的书房 vs 两开关的笔记本', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "LSTM vs GRU 门控对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 18:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('文本生成：学会了"语感"的学生', fontsize=16, fontweight='bold')
ax.text(6, 7, '输入: "深度学习"', fontsize=14, ha='center', fontweight='bold', color='#2c3e50',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3'))
temps = [('温度=0.5\\n保守', 2, 4, '#3498db', '深度学习是人工智能...'),
         ('温度=1.0\\n平衡', 6, 4, '#27ae60', '深度学习改变了世界...'),
         ('温度=1.5\\n创造', 10, 4, '#e74c3c', '深度学习将引领未来...')]
for label, x, y, color, text in temps:
    ax.text(x, y+1, label, fontsize=12, ha='center', fontweight='bold', color=color)
    ax.text(x, y-0.5, text, fontsize=10, ha='center', color='#555',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.1))
ax.annotate('', xy=(6, 6.3), xytext=(6, 5.5), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
ax.text(6, 1.5, '温度越低越保守(重复训练文本)\\n温度越高越创造(可能胡言乱语)', fontsize=12, ha='center',
       color='#2c3e50', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f'))
plt.tight_layout()
plt.show()
""", "文本生成温度控制示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 19:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Seq2Seq: 同声传译的两步走', fontsize=16, fontweight='bold')
ax.add_patch(plt.Rectangle((0.5, 5), 5, 3.5, facecolor='#d5f5e3', edgecolor='#27ae60', lw=3))
ax.text(3, 8, 'Encoder (听)', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
src_words = ['I', 'love', 'deep', 'learning']
for i, w in enumerate(src_words):
    ax.text(1.5 + i*1.1, 6.5, w, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', edgecolor='#27ae60'))
ax.add_patch(plt.Rectangle((5.5, 6), 3, 1.5, facecolor='#f9e79f', edgecolor='#f39c12', lw=3))
ax.text(7, 6.75, '上下文向量\\n[0.3, -0.7, ...]', fontsize=10, ha='center', fontweight='bold', color='#f39c12')
ax.annotate('', xy=(5.5, 6.75), xytext=(5.5, 6.75), arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))
ax.add_patch(plt.Rectangle((8.5, 5), 5, 3.5, facecolor='#d4e6f1', edgecolor='#2980b9', lw=3))
ax.text(11, 8, 'Decoder (说)', fontsize=14, ha='center', fontweight='bold', color='#2980b9')
tgt_words = ['我', '爱', '深度', '学习']
for i, w in enumerate(tgt_words):
    ax.text(9.5 + i*1.1, 6.5, w, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#ebf5fb', edgecolor='#2980b9'))
ax.text(7, 3, '瓶颈: 整个句子压缩成\\n一个固定向量\\n长句子信息丢失!', fontsize=12, ha='center',
       color='#e74c3c', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0'))
ax.text(7, 1, '解决方案: 注意力机制 (明天预告!)', fontsize=13, ha='center',
       color='#27ae60', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
plt.tight_layout()
plt.show()
""", "Seq2Seq 信息瓶颈示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 20:
            if '## 4. 阶段五总结' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('阶段五知识演进路线图', fontsize=16, fontweight='bold')
topics = [
    (5, 11, '#e74c3c', 'Day 01: 金鱼记忆', '前馈网络没有记忆'),
    (5, 9.8, '#f39c12', 'Day 02: RNN 隐藏状态', 'h_t = tanh(W*x_t + W*h_{t-1} + b)'),
    (5, 8.6, '#e67e22', 'Day 03: 梯度消失', 'RNN 只能记住最近几步'),
    (5, 7.4, '#27ae60', 'Day 04: LSTM 三个门', '遗忘+输入+输出+细胞状态'),
    (5, 6.2, '#2980b9', 'Day 05: LSTM vs RNN', '加法更新碾压乘法链'),
    (5, 5.0, '#9b59b6', 'Day 06: 词嵌入', '文字变成有意义的向量'),
    (5, 3.8, '#1abc9c', 'Day 07: GRU', '两个门就够了'),
    (5, 2.6, '#e74c3c', 'Day 08: 文本生成', '逐字预测+温度控制'),
    (5, 1.4, '#3498db', 'Day 09: Seq2Seq', 'Encoder-Decoder+信息瓶颈'),
    (5, 0.2, '#2c3e50', 'Day 10: 收官实战', '姓名国籍分类器'),
]
for x, y, color, title, desc in topics:
    ax.add_patch(plt.Rectangle((0.5, y-0.35), 9, 0.7, facecolor=color, alpha=0.12, edgecolor=color, lw=1.5))
    ax.text(1, y, title, fontsize=10, fontweight='bold', color=color, va='center')
    ax.text(4.5, y, desc, fontsize=9, color='#555', va='center')
for i in range(len(topics)-1):
    ax.annotate('', xy=(5, topics[i+1][1]+0.35), xytext=(5, topics[i][1]-0.35),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1))
plt.tight_layout()
plt.show()
""", "阶段五知识路线图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 21:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('Seq2Seq: 凭记忆翻译', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
src = ['The', 'cat', 'sat', 'on', 'the', 'mat']
for i, w in enumerate(src):
    alpha = max(0.2, 1.0 - i*0.15)
    ax.text(1 + i*1.3, 6, w, fontsize=10, ha='center', alpha=alpha,
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=alpha))
ax.text(5, 3.5, '压缩向量\\n(信息丢失!)', fontsize=12, ha='center', color='#e74c3c', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax.text(5, 1.5, '翻译官凭记忆翻译\\n长句子就忘了', fontsize=11, ha='center', color='#555')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('注意力: 翻译时回头看', fontsize=14, fontweight='bold', color='#27ae60')
ax.axis('off')
for i, w in enumerate(src):
    ax.text(1 + i*1.3, 6, w, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5f5e3', edgecolor='#27ae60'))
ax.text(5, 3.5, 'Q: "我要翻译什么?"\\nK: "源句子有什么?"\\nV: "源句子的内容"', fontsize=11, ha='center',
       color='#27ae60', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
ax.text(5, 1.5, '翻译每个词时回头看原文\\n动态聚焦最相关部分', fontsize=11, ha='center', color='#555')
plt.suptitle('信息瓶颈 vs 注意力机制', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "Seq2Seq瓶颈 vs 注意力机制对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 22:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis('off')
ax.set_title('Q、K、V 三角舞：图书馆找书', fontsize=16, fontweight='bold')
ax.add_patch(plt.Circle((3, 6), 1.2, facecolor='#e74c3c', alpha=0.2, edgecolor='#e74c3c', lw=2))
ax.text(3, 6, 'Q\\n查询\\n"我要找\\n深度学习"', fontsize=10, ha='center', fontweight='bold', color='#e74c3c')
ax.add_patch(plt.Circle((9, 6), 1.2, facecolor='#27ae60', alpha=0.2, edgecolor='#27ae60', lw=2))
ax.text(9, 6, 'K\\n键\\n"每本书\\n的标签"', fontsize=10, ha='center', fontweight='bold', color='#27ae60')
ax.add_patch(plt.Circle((6, 2), 1.2, facecolor='#3498db', alpha=0.2, edgecolor='#3498db', lw=2))
ax.text(6, 2, 'V\\n值\\n"书的\\n实际内容"', fontsize=10, ha='center', fontweight='bold', color='#3498db')
ax.annotate('', xy=(7.8, 6), xytext=(4.2, 6), arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))
ax.text(6, 6.5, 'Q·K = 匹配度', fontsize=11, ha='center', color='#f39c12', fontweight='bold')
ax.annotate('', xy=(6, 3.2), xytext=(6, 4.8), arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2))
ax.text(7.5, 4, 'softmax(QK^T/√d)\\n× V = 上下文', fontsize=11, ha='center', color='#9b59b6', fontweight='bold')
plt.tight_layout()
plt.show()
""", "QKV 三角舞示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 23:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('多头注意力：多个侦探同时调查', fontsize=16, fontweight='bold')
detectives = [
    (2.5, 5, '#e74c3c', '侦探A', '查时间线\\n(语法关系)'),
    (6, 5, '#27ae60', '侦探B', '查人际关系\\n(语义关系)'),
    (9.5, 5, '#3498db', '侦探C', '查动机\\n(位置关系)'),
    (13, 5, '#f39c12', '侦探D', '查证据\\n(其他模式)'),
]
for x, y, color, name, desc in detectives:
    ax.add_patch(plt.Circle((x, y), 1, facecolor=color, alpha=0.15, edgecolor=color, lw=2))
    ax.text(x, y+0.3, name, fontsize=11, ha='center', fontweight='bold', color=color)
    ax.text(x, y-0.4, desc, fontsize=9, ha='center', color='#555')
ax.add_patch(plt.Rectangle((1, 1), 12, 2, facecolor='#f9e79f', edgecolor='#f39c12', lw=2))
ax.text(7, 2, '汇总报告 = Concat(头1, 头2, 头3, 头4) × W_O', fontsize=13, ha='center', fontweight='bold', color='#f39c12')
for x, _, color, _, _ in detectives:
    ax.annotate('', xy=(7, 3), xytext=(x, 4), arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))
plt.tight_layout()
plt.show()
""", "多头注意力侦探隐喻图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 24:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('位置编码：座次牌', fontsize=16, fontweight='bold')
positions = ['词1', '词2', '词3', '词4', '词5']
for i, pos in enumerate(positions):
    x = 1.5 + i * 2
    ax.add_patch(plt.Rectangle((x-0.7, 4), 1.4, 2, facecolor='#d5f5e3', edgecolor='#27ae60', lw=2))
    ax.text(x, 5.5, pos, fontsize=12, ha='center', fontweight='bold', color='#27ae60')
    ax.add_patch(plt.Rectangle((x-0.5, 3), 1, 0.8, facecolor='#f9e79f', edgecolor='#f39c12', lw=1.5))
    ax.text(x, 3.4, f'位置{i+1}', fontsize=10, ha='center', fontweight='bold', color='#f39c12')
    if i > 0:
        ax.annotate('', xy=(x-0.8, 5), xytext=(x-1.2, 5), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(6, 1.5, '没有座次牌 → 不知道谁在哪儿 (注意力无顺序)\\n有座次牌 → 知道每个词的位置 (正弦位置编码)',
       fontsize=12, ha='center', color='#2c3e50', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf2f8'))
plt.tight_layout()
plt.show()
""", "位置编码座次牌示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 25:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Transformer Encoder: 团队会议决策流程', fontsize=16, fontweight='bold')
steps = [
    (7, 9, 10, 0.8, '#3498db', '输入: 词嵌入 + 位置编码'),
    (7, 7.8, 10, 0.8, '#27ae60', '多头自注意力: 每个人听所有人发言'),
    (7, 6.6, 10, 0.8, '#e74c3c', '残差连接 + 层归一化: 保留自己的想法 + 统一音量'),
    (7, 5.4, 10, 0.8, '#9b59b6', '前馈网络: 每个人独立思考'),
    (7, 4.2, 10, 0.8, '#e74c3c', '残差连接 + 层归一化: 保留思考结果 + 统一输出'),
    (7, 3, 10, 0.8, '#f39c12', '输出: 每个词都融入了上下文信息'),
]
for x, y, w, h, color, label in steps:
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, alpha=0.15, edgecolor=color, lw=2))
    ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold', color=color)
for i in range(len(steps)-1):
    ax.annotate('', xy=(7, steps[i+1][1]+0.4), xytext=(7, steps[i][1]-0.4),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(7, 1.5, '关键: 残差连接让信息保底, 层归一化让训练稳定', fontsize=12, ha='center',
       color='#e74c3c', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
plt.tight_layout()
plt.show()
""", "Transformer Encoder 决策流程图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 26:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Transformer Decoder: 带参考书的闭卷考试', fontsize=16, fontweight='bold')
steps = [
    (7, 9, 10, 0.7, '#3498db', '输入: 已生成的词 + 位置编码'),
    (7, 8, 10, 0.7, '#27ae60', '掩码自注意力: 只看已写的内容 (因果掩码)'),
    (7, 7, 10, 0.7, '#e74c3c', '残差 + 层归一化'),
    (7, 6, 10, 0.7, '#9b59b6', '交叉注意力: Q来自Decoder, K/V来自Encoder (翻看原文!)'),
    (7, 5, 10, 0.7, '#e74c3c', '残差 + 层归一化'),
    (7, 4, 10, 0.7, '#f39c12', '前馈网络: 独立思考'),
    (7, 3, 10, 0.7, '#e74c3c', '残差 + 层归一化'),
    (7, 2, 10, 0.7, '#2c3e50', '输出: 下一个词的概率分布'),
]
for x, y, w, h, color, label in steps:
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, alpha=0.12, edgecolor=color, lw=1.5))
    ax.text(x, y, label, fontsize=9, ha='center', va='center', fontweight='bold', color=color)
for i in range(len(steps)-1):
    ax.annotate('', xy=(7, steps[i+1][1]+0.35), xytext=(7, steps[i][1]-0.35),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1))
plt.tight_layout()
plt.show()
""", "Transformer Decoder 流程图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 27:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('完整 Transformer 架构', fontsize=16, fontweight='bold')
ax.add_patch(plt.Rectangle((0.5, 1), 5, 8, facecolor='#d5f5e3', edgecolor='#27ae60', lw=3))
ax.text(3, 8.5, 'Encoder', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
enc_steps = ['词嵌入+位置编码', '多头自注意力', '残差+归一化', '前馈网络', '残差+归一化']
for i, s in enumerate(enc_steps):
    ax.text(3, 7 - i*1.3, s, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', edgecolor='#27ae60'))
ax.add_patch(plt.Rectangle((8.5, 1), 5, 8, facecolor='#d4e6f1', edgecolor='#2980b9', lw=3))
ax.text(11, 8.5, 'Decoder', fontsize=14, ha='center', fontweight='bold', color='#2980b9')
dec_steps = ['词嵌入+位置编码', '掩码自注意力', '交叉注意力', '前馈网络', '线性+Softmax']
for i, s in enumerate(dec_steps):
    ax.text(11, 7 - i*1.3, s, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#ebf5fb', edgecolor='#2980b9'))
ax.annotate('', xy=(8.5, 4.5), xytext=(5.5, 4.5), arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
ax.text(7, 5, 'K, V', fontsize=12, ha='center', fontweight='bold', color='#e74c3c')
plt.tight_layout()
plt.show()
""", "完整 Transformer 架构图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 28:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('BERT: 阅读理解考官', fontsize=14, fontweight='bold', color='#3498db')
ax.axis('off')
ax.text(5, 6.5, 'Encoder-only', fontsize=14, ha='center', fontweight='bold', color='#3498db')
ax.text(5, 5, '双向注意力\\n每个词看到前后所有词', fontsize=12, ha='center', color='#555')
ax.text(5, 3.5, '任务: 完形填空\\n"我[MASK]深度学习"', fontsize=12, ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', edgecolor='#3498db'))
ax.text(5, 1.5, '擅长: 理解、分类、问答', fontsize=12, ha='center', color='#3498db', fontweight='bold')
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title('GPT: 创意写作作家', fontsize=14, fontweight='bold', color='#e74c3c')
ax.axis('off')
ax.text(5, 6.5, 'Decoder-only', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')
ax.text(5, 5, '单向注意力\\n每个词只看到前面的词', fontsize=12, ha='center', color='#555')
ax.text(5, 3.5, '任务: 预测下一个词\\n"我爱深度→[学习]"', fontsize=12, ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#e74c3c'))
ax.text(5, 1.5, '擅长: 生成、对话、翻译', fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
plt.suptitle('BERT vs GPT: 考官 vs 作家', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
""", "BERT vs GPT 对比图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 29:
            if '## 2. 生活隐喻' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('学习率预热: 开车上高速', fontsize=16, fontweight='bold')
steps_lr = np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0.3, 30)])
ax.plot(range(len(steps_lr)), steps_lr, color='#e74c3c', lw=3)
ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
ax.text(10, 1.05, 'Warmup结束', fontsize=10, color='gray')
ax.text(10, 0.7, '慢慢加速', fontsize=12, ha='center', color='#27ae60', fontweight='bold')
ax.text(35, 0.7, '逐渐减速', fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
ax.fill_between(range(20), steps_lr[:20], alpha=0.2, color='#27ae60')
ax.fill_between(range(20, 50), steps_lr[20:], alpha=0.2, color='#e74c3c')
ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel('学习率', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""", "学习率预热示意图")
                if dc: visual_cells.append(dc)

        elif chapter_num == 30:
            if '## 3. 核心洞察' in source_text or '## 4. 核心洞察' in source_text:
                dc = make_diagram_md_cell("""
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('从感知机到 Transformer: 12周觉醒之路', fontsize=16, fontweight='bold')
eras = [
    (7, 11, 12, 0.8, '#e74c3c', '阶段一: AI初春与寒冬 (Week 1-2) — 感知机 → XOR危机 → MLP'),
    (7, 9.8, 12, 0.8, '#f39c12', '阶段二: 破局与复兴 (Week 3-4) — 激活函数 → 反向传播 → 梯度下降'),
    (7, 8.6, 12, 0.8, '#27ae60', '阶段三: 视觉征服 (Week 5-6) — CNN → 卷积核 → 梯度消失'),
    (7, 7.4, 12, 0.8, '#2980b9', '阶段四: ResNet奇迹 (Week 7-8) — 退化问题 → 残差连接 → F(x)+x'),
    (7, 6.2, 12, 0.8, '#9b59b6', '阶段五: 记忆诞生 (Week 9-10) — RNN → LSTM → Seq2Seq'),
    (7, 5.0, 12, 0.8, '#e67e22', '阶段六: 注意力时代 (Week 11-12) — QKV → 多头 → Transformer'),
]
for x, y, w, h, color, label in eras:
    ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, alpha=0.15, edgecolor=color, lw=2))
    ax.text(x, y, label, fontsize=9, ha='center', va='center', fontweight='bold', color=color)
for i in range(len(eras)-1):
    ax.annotate('', xy=(7, eras[i+1][1]+0.4), xytext=(7, eras[i][1]-0.4),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(7, 3, '三条核心线索:', fontsize=13, ha='center', fontweight='bold', color='#2c3e50')
ax.text(3, 2, '信息流动\\n单向→反向→抄近道→自由流动', fontsize=10, ha='center', color='#e74c3c',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0'))
ax.text(7, 2, '梯度问题\\n消失→残差保底→门控保底', fontsize=10, ha='center', color='#27ae60',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0'))
ax.text(11, 2, '视野范围\\n局部→序列→全局注意力', fontsize=10, ha='center', color='#2980b9',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8'))
ax.text(7, 0.5, '深度学习的进化 = 不断解决"信息如何更好地流动"', fontsize=13, ha='center',
       color='#2c3e50', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f9e79f'))
plt.tight_layout()
plt.show()
""", "12周觉醒之路全景图")
                if dc: visual_cells.append(dc)

    return visual_cells


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python",
                "name": "python", "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3", "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


chapter_files = [
    ("Week07/chapter_01_paradox_of_depth.md", "Week07/Day01_深度悖论_层数越多反而越蠢.ipynb", 1),
    ("Week07/chapter_02_degradation_code_evidence.md", "Week07/Day02_退化问题的代码实证.ipynb", 2),
    ("Week07/chapter_03_residual_connection_intuition.md", "Week07/Day03_残差连接的直觉_抄近道的艺术.ipynb", 3),
    ("Week07/chapter_04_building_residual_block.md", "Week07/Day04_手写残差块_Fx_plus_x的代码奇迹.ipynb", 4),
    ("Week07/chapter_05_plain_vs_resnet_battle.md", "Week07/Day05_史诗对决_Plain_Net_vs_ResNet.ipynb", 5),
    ("Week08/chapter_06_gradient_highway.md", "Week08/Day06_梯度高速公路_为什么跳跃连接有效.ipynb", 6),
    ("Week08/chapter_07_identity_mapping.md", "Week08/Day07_恒等映射的完美_当Fx等于0.ipynb", 7),
    ("Week08/chapter_08_from_block_to_network.md", "Week08/Day08_从块到网络_堆叠残差块的艺术.ipynb", 8),
    ("Week08/chapter_09_resnet_family.md", "Week08/Day09_ResNet家族_从18到152.ipynb", 9),
    ("Week08/chapter_10_build_mini_resnet.md", "Week08/Day10_收官之战_从零搭建迷你ResNet.ipynb", 10),
    ("Week09/chapter_01_goldfish_memory.md", "Week09/Day01_金鱼记忆_为什么前馈网络记不住.ipynb", 11),
    ("Week09/chapter_02_rnn_hidden_state.md", "Week09/Day02_隐藏状态_RNN边看边记的秘密.ipynb", 12),
    ("Week09/chapter_03_rnn_vanishing_gradient.md", "Week09/Day03_梯度消失_RNN的致命缺陷.ipynb", 13),
    ("Week09/chapter_04_lstm_gates.md", "Week09/Day04_LSTM三个门_遗忘输入输出的魔法.ipynb", 14),
    ("Week09/chapter_05_rnn_vs_lstm_battle.md", "Week09/Day05_金鱼vs大象_RNN与LSTM的终极对决.ipynb", 15),
    ("Week10/chapter_06_word_embedding.md", "Week10/Day06_词嵌入_给文字发身份证.ipynb", 16),
    ("Week10/chapter_07_gru.md", "Week10/Day07_GRU_LSTM的精简版表弟.ipynb", 17),
    ("Week10/chapter_08_text_generation.md", "Week10/Day08_文本生成_学会了语感的学生.ipynb", 18),
    ("Week10/chapter_09_seq2seq.md", "Week10/Day09_Seq2Seq_同声传译的两步走.ipynb", 19),
    ("Week10/chapter_10_lstm_final_project.md", "Week10/Day10_收官之战_姓名国籍分类器.ipynb", 20),
    ("Week11/chapter_01_attention_intuition.md", "Week11/Day01_注意力的直觉_回头看的力量.ipynb", 21),
    ("Week11/chapter_02_qkv_attention.md", "Week11/Day02_QKV机制_图书馆找书的三角舞.ipynb", 22),
    ("Week11/chapter_03_multi_head_attention.md", "Week11/Day03_多头注意力_多个侦探同时调查.ipynb", 23),
    ("Week11/chapter_04_positional_encoding.md", "Week11/Day04_位置编码_给词语发座次牌.ipynb", 24),
    ("Week11/chapter_05_transformer_encoder.md", "Week11/Day05_Transformer_Encoder_团队会议决策.ipynb", 25),
    ("Week12/chapter_06_transformer_decoder.md", "Week12/Day06_Transformer_Decoder_带参考书的闭卷考试.ipynb", 26),
    ("Week12/chapter_07_full_transformer.md", "Week12/Day07_完整Transformer_从图纸到现实.ipynb", 27),
    ("Week12/chapter_08_bert_vs_gpt.md", "Week12/Day08_BERT_vs_GPT_考官与作家.ipynb", 28),
    ("Week12/chapter_09_training_tricks.md", "Week12/Day09_训练技巧_让Transformer跑起来.ipynb", 29),
    ("Week12/chapter_10_final_transformer.md", "Week12/Day10_收官之战_从感知机到Transformer的觉醒之路.ipynb", 30),
]

for md_path, ipynb_path, chapter_num in chapter_files:
    print(f"Converting: {os.path.basename(md_path)} -> {os.path.basename(ipynb_path)}")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    cells = parse_md_to_cells(md_content)
    cells_with_diagrams = add_visual_diagrams(cells, chapter_num)
    notebook = create_notebook(cells_with_diagrams)
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    md_cells = sum(1 for c in cells_with_diagrams if c['cell_type'] == 'markdown')
    code_cells = sum(1 for c in cells_with_diagrams if c['cell_type'] == 'code')
    img_cells = sum(1 for c in cells_with_diagrams if c['cell_type'] == 'markdown' and any('data:image/png' in s for s in c.get('source', [])))
    print(f"  Done! {md_cells} md cells ({img_cells} with images), {code_cells} code cells")

print("\nAll conversions complete!")
