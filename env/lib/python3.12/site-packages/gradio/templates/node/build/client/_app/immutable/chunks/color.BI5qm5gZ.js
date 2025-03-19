import { o as ordered_colors } from "./2.Z6KyGJbu.js";
const get_next_color = (index) => {
  return ordered_colors[index % ordered_colors.length];
};
export {
  get_next_color as g
};
