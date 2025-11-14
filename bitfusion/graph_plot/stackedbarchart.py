#!/usr/bin/python
# python bitfusion/graph_plot/stackedbarchart.py --csv_file results/ant_res.csv --output_file results/ant_res.pdf
import os,sys,math,csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
import argparse

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
COLOR = []

ANT_COMPONENTS = ['Static', 'Dram', 'Buffer', 'Core']

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    s = s.split('.')[0]

    # The percent symbol needs escaping in latex
    if rcParams['text.usetex'] == True:
        return r'\sf{%s%s}' % (s, r'\%')
    else:
        return s + '%'
def to_fixed(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations
    s = str(y)
    return s

def change_font(y, position):
	x = int(y)
	if rcParams['text.usetex'] == True:
		return r'\sf{%s}' % x

def autolabel(ax, rects, tlMargin):
	for i in range(0, len(rects)):
		rect = rects[i]
		height = rect.get_height()
		if height > YAXIS_MAX:
			ax.text(rect.get_x() + rect.get_width()/2. + tlMargin, YAXIS_MAX * 1.01, TOPLABEL_FORMAT%float(height), ha='center', va='bottom', fontsize=TOP_LABEL_FONTSIZE, rotation=TOPLABEL_ROTATE, fontweight=TOPLABEL_BOLD)

def datalabel(ax, rects, tlMargin):
	for i in range(0, len(rects)):
		rect = rects[i]
		height = rect.get_height()
		# change to move slightly right for num-annotations
		minor = 0.0
		if height == 11:
			minor = 0.07
		if height == 40:
			minor = 0.11
		if height == 14:
			minor = 0.07
		if height == 109:
			minor = 0.16
		if height == 39:
			minor = 0.09
		ax.text(rect.get_x() + rect.get_width()/2. + tlMargin - 0.02 + minor, float(height) + YAXIS_MAX * 0.01, DATALABEL_FORMAT%int(height), ha='center', va='bottom', fontsize=DATA_LABEL_FONTSIZE, rotation=DATALABEL_ROTATE, fontweight=DATALABEL_BOLD)


def _looks_like_ant_results(csvFilename):
	try:
		with open(csvFilename, 'r') as f:
			first = f.readline()
			second = f.readline()
			third = f.readline()
	except IOError:
		return False
	if not first:
		return False
	first = first.strip()
	second = second.strip()
	third = third.strip()
	return first.startswith(',') and second.startswith(',') and third.lower().startswith('time')

def _extract_bench_metadata(header_row):
	bench_names = []
	column_bench = []
	current = None
	for cell in header_row[1:]:
		name = cell.strip()
		if name:
			current = name
			if not bench_names or bench_names[-1] != name:
				bench_names.append(name)
		column_bench.append(current)
	return bench_names, column_bench

def _extract_arch_metadata(arch_row):
	architectures = []
	column_arch = []
	for cell in arch_row[1:]:
		name = cell.strip()
		if name and name not in architectures:
			architectures.append(name)
		column_arch.append(name.strip())
	return architectures, column_arch

def read_ant_results(csvFilename):
	with open(csvFilename, 'r') as f:
		reader = csv.reader(f, skipinitialspace=True)
		rows = [row for row in reader]

	if len(rows) < 9:
		raise RuntimeError('Unexpected ANT results CSV layout: need at least 9 rows')

	header_row = rows[0]
	arch_row = rows[1]
	time_row = rows[2]

	component_rows = [row for row in rows if row and row[0].strip() in ANT_COMPONENTS]

	bench_names, column_bench = _extract_bench_metadata(header_row)
	architectures, column_arch = _extract_arch_metadata(arch_row)

	bench_to_index = {name: idx for idx, name in enumerate(bench_names)}

	cycle_data = {arch: [None]*len(bench_names) for arch in architectures}

	for bench_name, arch_name, value in zip(column_bench, column_arch, time_row[1:]):
		if not bench_name or not arch_name:
			continue
		val = value.strip()
		if not val:
			continue
		cycle_data[arch_name][bench_to_index[bench_name]] = float(val)

	energy_data = {}
	for comp_row in component_rows:
		component = comp_row[0].strip()
		if component not in ANT_COMPONENTS:
			continue
		energy_data.setdefault(component, {arch: [None]*len(bench_names) for arch in architectures})
		for bench_name, arch_name, value in zip(column_bench, column_arch, comp_row[1:]):
			if not bench_name or not arch_name:
				continue
			val = value.strip()
			if not val:
				continue
			energy_data[component][arch_name][bench_to_index[bench_name]] = float(val)

	return bench_names, architectures, cycle_data, energy_data


def _str_to_bool(value):
	if isinstance(value, bool):
		return value
	val = value.lower()
	if val in ('true', 't', '1', 'yes', 'y'):
		return True
	if val in ('false', 'f', '0', 'no', 'n'):
		return False
	raise argparse.ArgumentTypeError("Expected a boolean flag (true/false)")


def _rotation_value(value):
	val = value.lower()
	if val == 'false':
		return 0.0
	try:
		return float(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("Expected a float or 'False'") from exc


def _color_gradient(count, start=None, end=None):
	if count <= 0:
		return []
	default_min = 0.15 if COLOR_MIN is None else COLOR_MIN
	default_max = 0.95 if COLOR_MAX is None else COLOR_MAX
	start = default_min if start is None else start
	end = default_max if end is None else end
	if count == 1:
		return [str(end)]
	step = (end - start) / float(count - 1) if count > 1 else 0.0
	return [str(end - step * idx) for idx in range(count)]

def plot_ant_results(outputFileName, bench_names, architectures, cycle_data, energy_data):
	n_bench = len(bench_names)
	n_arch = len(architectures)
	bar_width = BAR_WIDTH
	inner_gap = 0.1	# 每组内两根柱子之间的缝隙
	group_width = (bar_width + inner_gap) * n_arch + 0.2
	group_centers = np.arange(n_bench) * group_width
	offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * (bar_width + inner_gap)

	arch_colors = ['#4C72B0', '#FFFFFF', '#4AC0E0', '#FFBF00', '#D62728', '#B5D68A']
	component_colors = {}
	fallback_palette = _color_gradient(
		len(ANT_COMPONENTS),
		start=COLOR_MIN,
		end=min(COLOR_MAX if COLOR_MAX is not None else 0.95, 0.85)
	)
	for idx, comp in enumerate(ANT_COMPONENTS):
		if idx < len(arch_colors):
			component_colors[comp] = arch_colors[idx]
		else:
			component_colors[comp] = fallback_palette[idx]
	component_styles = {
		'Static': {'color': arch_colors[0], 'edgecolor': 'black', 'hatch': '', 'lw': 0.25},
		'Dram': {'color': arch_colors[1], 'edgecolor': 'black', 'hatch': '///', 'lw': 0.25},
		'Buffer': {'color': arch_colors[2], 'edgecolor': 'black', 'hatch': '..', 'lw': 0.6},
		'Core': {'color': arch_colors[3], 'edgecolor': 'black', 'hatch': '', 'lw': 0.25},
	}

	fig_height = FIG_HEIGHT if FIG_HEIGHT is not None else 3.5
	fig_width = FIG_WIDTH if FIG_WIDTH is not None else 8.0
	fig, (ax_time, ax_energy) = plt.subplots(
		2, 1, sharex=True,
		figsize=(fig_width, fig_height * 1.8),
		gridspec_kw={'height_ratios': [1.0, 1.2]}
	)

	if GLOBAL_FONTSIZE is not None:
		rcParams['font.size'] = GLOBAL_FONTSIZE

	geo_index = None
	for idx, name in enumerate(bench_names):
		if name.strip().lower() in ('geomean', 'gmean', 'geomean.', 'avg', 'average'):
			geo_index = idx
			break

	# 开始绘制第一张子图: Time
	max_time = 0.0
	for idx, arch in enumerate(architectures):
		values = np.array([
			cycle_data.get(arch, [None]*n_bench)[i]
			if cycle_data.get(arch, [None]*n_bench)[i] is not None else np.nan
			for i in range(n_bench)
		], dtype=float)
		mask = ~np.isnan(values)
		if not np.any(mask):
			continue
		max_time = max(max_time, np.nanmax(values))
		hatch_styles = ['', '///', '..', '', '', '']
		edge_colors = ['black', 'black', 'black', 'black', 'black', 'black']
		edge_color = edge_colors[idx] if idx < len(edge_colors) else 'black'
		ax_time.bar(
			group_centers[mask] + offsets[idx],
			values[mask],
			width=bar_width,
			color=arch_colors[idx],
			edgecolor=edge_color,
			label=arch,
			linewidth=0.25,
			hatch=hatch_styles[idx],
		)
		# 对于geomean这个组，展示数据
		if geo_index is not None and geo_index < len(values):
			val = values[geo_index]
			if not np.isnan(val) and val != 1.0:
				offset = max_time * 0.02 if max_time > 0 else 0.02
				x_pos = group_centers[geo_index] + offsets[idx]
				ax_time.text(x_pos, val + offset, f'{val:.2f}',
				             ha='center', va='bottom', fontsize=DATA_LABEL_FONTSIZE)

	ax_time.set_ylabel('Norm. Cycle', fontsize=AXIS_TITLE_FONTSIZE)
	if max_time > 0:
		ax_time.set_ylim(0, max_time)
	ax_time.grid(axis='y', color='0.85')
	ax_time.set_axisbelow(True)

	# 开始绘制第二张子图：Energy Breakdown
	for idx, arch in enumerate(architectures):
		bottom = np.zeros(n_bench)
		for component in ANT_COMPONENTS:
			comp_map = energy_data.get(component, {})
			values_list = comp_map.get(arch)
			if not values_list:
				continue
			values = np.array([
				val if val is not None else 0.0 for val in values_list
			], dtype=float)
			mask = np.array([val is not None for val in values_list])
			if not np.any(mask):
				continue
			style = component_styles.get(component, {})
			color = style.get('color', component_colors.get(component, str(COLOR_MIN)))
			edgecolor = style.get('edgecolor', 'black')
			hatch = style.get('hatch', '')
			linewidth = style.get('lw', 0.25)
			ax_energy.bar(
				group_centers[mask] + offsets[idx],
				values[mask],
				width=bar_width,
				bottom=bottom[mask],
				color=color,
				label=component if idx == 0 else None,
				edgecolor=edgecolor,
				linewidth=linewidth,
				hatch=hatch
			)
			bottom[mask] += values[mask]
		# 对于geomean这个组，展示数据
		if geo_index is not None and geo_index < len(bottom):
			total = bottom[geo_index]
			if not np.isnan(total) and total != 1.0 and total > 0:
				offset = total * 0.02
				x_pos = group_centers[geo_index] + offsets[idx]
				ax_energy.text(x_pos, total + offset, f'{total:.2f}',
				               ha='center', va='bottom', fontsize=DATA_LABEL_FONTSIZE)

	comp_handles, comp_labels = ax_energy.get_legend_handles_labels()
	if comp_handles:
		ax_energy.legend(
			comp_handles,
			comp_labels,
			ncol=len(comp_labels),
			fontsize=LEGEND_FONTSIZE,
			loc='lower center',
			bbox_to_anchor=(0.5, 0.95),
			frameon=False
		)

	# 调整画布的左右边界，让柱状图紧贴画布的左右边缘。
	x_min = group_centers[0] + offsets[0] - bar_width / 2.0 - 0.2
	x_max = group_centers[-1] + offsets[-1] + bar_width / 2.0 +0.2
	for axis in (ax_time, ax_energy):
		axis.set_xlim(x_min, x_max)

	ax_energy.set_ylabel('Norm. Energy', fontsize=AXIS_TITLE_FONTSIZE)
	ax_energy.grid(axis='y', color='0.85')
	ax_energy.set_axisbelow(True)
	ax_energy.set_xticks(group_centers)
	ax_energy.set_xticklabels([''] * len(bench_names))
	xaxis_transform = ax_energy.get_xaxis_transform()
	# 绘制不同benchmark的标签
	arch_label_y = -0.03
	bench_label_y = -0.38
	for bench_idx, bench_name in enumerate(bench_names):
		center = group_centers[bench_idx]
		ax_energy.text(center, bench_label_y, bench_name, ha='center', va='top',
			fontsize=XAXIS_FONTSIZE, rotation=ROTATEXAXIS,
			transform=xaxis_transform)
		for arch_idx, arch in enumerate(architectures):
			if arch_idx >= len(offsets):
				continue
			xpos = center + offsets[arch_idx]
			ax_energy.text(xpos, arch_label_y, arch, ha='center', va='top',
				fontsize=XAXIS_FONTSIZE, rotation=90,
				transform=xaxis_transform)
	# 绘制不同benchmark之间的分隔竖线
	seperator_top = - 0.02
	seperator_bottom = bench_label_y - 0.02
	x_sep_0 = group_centers[0] - 0.5*group_width
	ax_energy.plot([x_sep_0, x_sep_0], [seperator_top, seperator_bottom], transform=xaxis_transform, color='gray', linewidth=0.5, clip_on=False)
	for idx in range(len(group_centers)):
		x_sep = group_centers[idx] + 0.5*group_width
		ax_energy.plot([x_sep, x_sep], [seperator_top, seperator_bottom], transform=xaxis_transform, color='gray', linewidth=0.5, clip_on=False)

	for axis in (ax_time, ax_energy):
		axis.tick_params(axis='y', pad=YAXIS_PAD if YAXIS_PAD is not None else 10)

	bottom_pad = XAXISPADDING if XAXISPADDING is not None else 0.3
	bottom_pad = max(bottom_pad, 0.4)
	fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=bottom_pad, hspace=0.1)

	with PdfPages(outputFileName) as pdf:
		pdf.savefig(fig, bbox_inches='tight', pad_inches=0.02)
	plt.close(fig)
	os.system('pdfcrop ' + outputFileName + ' ' + outputFileName + ' > /dev/null 2> /dev/null')

def main():

	global  YAXIS_MIN, YAXIS_MAX, FIG_WIDTH, FIG_HEIGHT, BAR_WIDTH, COLOR_MIN, COLOR_MAX, GLOBAL_FONTSIZE, LEGEND_FONTSIZE, \
			AXIS_TITLE_FONTSIZE, TOP_LABEL_FONTSIZE, XAXIS_FONTSIZE, YAXIS_PAD, \
			LEGEND_LOCATION, ROTATEXAXIS , XAXISPADDING, DATALABEL, DATA_LABEL_FONTSIZE, TOPLABEL_FORMAT, DATALABEL_FORMAT, \
			TOPLABEL_ROTATE, DATALABEL_ROTATE, TOPLABEL_BOLD, DATALABEL_BOLD, XAXIS_TEXTOFFSET, \
			BENCH_NEWLINE

	arg_parser = argparse.ArgumentParser(
		prog='stackedbarchart.py',
		description='Draw stacked bar charts from BitFusion CSV exports',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	arg_parser.add_argument('--csv_file', 									default="results/ant_res.csv",	help='Path to the source CSV file'				)
	arg_parser.add_argument('--output_file', 								default="results/ant_res.pdf",	help='Destination PDF file'						)
	arg_parser.add_argument('--ymin', 				type=float, 			default=0.0, 					help='Minimum value of Y-axis'					)
	arg_parser.add_argument('--ymax', 				type=float, 			default=6.0, 					help='Maximum value of Y-axis'					)
	arg_parser.add_argument('--fig_width', 			type=float, 			default=25.5, 					help='Figure width'								)
	arg_parser.add_argument('--fig_height', 		type=float, 			default=6, 						help='Figure height'							)
	arg_parser.add_argument('--bar_width', 			type=float, 			default=0.18, 					help='Width of each bar'						)
	arg_parser.add_argument('--color_min', 			type=float, 			default=0.15, 					help='Lightest bar color'						)
	arg_parser.add_argument('--color_max', 			type=float, 			default=0.95, 					help='Darkest bar color'						)
	arg_parser.add_argument('--top_label_fontsize', type=float, 			default=10.0, 					help='Font size for top labels'					)
	arg_parser.add_argument('--legend_fontsize', 	type=float, 			default=12.0, 					help='Font size for legend text'				)
	arg_parser.add_argument('--axis_title_fontsize',type=float, 			default=20.0, 					help='Font size for axis titles'				)
	arg_parser.add_argument('--global_fontsize', 	type=float, 			default=12.0, 					help='Global base font size'					)
	arg_parser.add_argument('--xaxis_fontsize', 	type=float, 			default=15.0, 					help='Font size for X-axis labels'				)
	arg_parser.add_argument('--yaxis_pad', 			type=float, 			default=10.0, 					help='Padding for Y-axis ticks'					)
	arg_parser.add_argument('--data_label_fontsize',type=float, 			default=10.0, 					help='Font size for data labels'				)
	arg_parser.add_argument('--rotate_xaxis', 		type=_rotation_value, 	default=0.0, 					help="Rotation angle for X-axis text or 'False'")
	arg_parser.add_argument('--legend_location', 	type=int, 				default=8, 						help='Matplotlib legend location code'			)
	arg_parser.add_argument('--xaxis_padding', 		type=float, 			default=0.4, 					help='Padding below X-axis labels'				)
	arg_parser.add_argument('--data_label', 		type=_str_to_bool, 		default=False,  				help='Display numeric labels on bars'			)
	arg_parser.add_argument('--data_label_format', 							default='%d', 					help='Format string for data labels'			)
	arg_parser.add_argument('--top_label_format', 							default='%.1f', 				help='Format string for top labels'				)
	arg_parser.add_argument('--top_label_rotate', 	type=int, 				default=0, 						help='Rotation for top labels'					)
	arg_parser.add_argument('--data_label_rotate', 	type=int, 				default=0, 						help='Rotation for data labels'					)
	arg_parser.add_argument('--xaxis_text_offset', 	type=float, 			default=0.5, 					help='Relative offset for X-axis text'			)
	arg_parser.add_argument('--top_label_bold', 							default='normal', 				help='Font weight for top labels'				)
	arg_parser.add_argument('--data_label_bold', 							default='normal', 				help='Font weight for data labels'				)
	arg_parser.add_argument('--bench_newline', 		type=_str_to_bool, 		default=False,  				help='Insert newline inside benchmark names'	)

	args = arg_parser.parse_args()

	csvFilename = args.csv_file
	outputFileName = args.output_file

	if os.path.isfile(csvFilename) == False:
		print( "Error: File [" + csvFilename + "] doesn't exist")
		sys.exit()

	if csvFilename[-4:] != ".csv":
		print( "Error: File [" + csvFilename + "] is not a csv file")
		sys.exit()

	YAXIS_MIN = args.ymin
	YAXIS_MAX = args.ymax
	FIG_WIDTH = args.fig_width
	FIG_HEIGHT = args.fig_height
	BAR_WIDTH = args.bar_width
	COLOR_MIN = args.color_min
	COLOR_MAX = args.color_max
	TOP_LABEL_FONTSIZE = args.top_label_fontsize
	LEGEND_FONTSIZE = args.legend_fontsize
	AXIS_TITLE_FONTSIZE = args.axis_title_fontsize
	GLOBAL_FONTSIZE = args.global_fontsize
	XAXIS_FONTSIZE = args.xaxis_fontsize
	YAXIS_PAD = args.yaxis_pad
	DATA_LABEL_FONTSIZE = args.data_label_fontsize
	ROTATEXAXIS = args.rotate_xaxis
	LEGEND_LOCATION = args.legend_location
	XAXISPADDING = args.xaxis_padding
	DATALABEL = args.data_label
	DATALABEL_FORMAT = args.data_label_format
	TOPLABEL_FORMAT = args.top_label_format
	TOPLABEL_ROTATE = args.top_label_rotate
	DATALABEL_ROTATE = args.data_label_rotate
	XAXIS_TEXTOFFSET = args.xaxis_text_offset
	TOPLABEL_BOLD = args.top_label_bold
	DATALABEL_BOLD = args.data_label_bold
	BENCH_NEWLINE = args.bench_newline

	if _looks_like_ant_results(csvFilename):
		bench_names, architectures, cycle_data, energy_data = read_ant_results(csvFilename)
		# 一共有9个bench，分别是：VGG16，ResNet18, ResNet50, Inception, ViT, BERT-MNLI, BERT-CoLA, BERT-SST-2, Geomean.
		# 一共有6个architectures，分别是：ANT-OS，ANT-WS, Bitfusion, OLAccel, BiScaled, AdaFloat.
		# cycle_data是一个字典，字典的每项包含一个key和一个对应的列表。key是架构名，列表是9个cycle数据。
		# energy_data也是一个字典，它有4个key，分别是：Static, Dram, Buffer, Core。字典的每项又是一个字典，包含六种架构分别在9个benchmark上的数据（如果某项数据为空，则用None填充）
		plot_ant_results(outputFileName, bench_names, architectures, cycle_data, energy_data)

if __name__ == "__main__":
    main()
