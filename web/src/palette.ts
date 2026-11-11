// Colours shared by the map and the panel.

/** One per candidate, in the order they are offered. */
export const ROUTE_COLOURS = ["#d7263d", "#1b6ef3", "#00916e", "#8e44ad", "#e67e22"];

/** Ridden roads by level, from rarely to often ridden (levels of GET /api/heatmap). */
export const HEAT_COLOURS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#a50026"];

export const routeColour = (index: number) => ROUTE_COLOURS[index % ROUTE_COLOURS.length]!;
