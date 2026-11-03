/* Shared course geometry for the keyboard and USB editions. Units are metres. */
(function (root) {
  'use strict';
  const ranks = ['新手', '進階', '高手'];
  const numerals = ['一', '二', '三'];
  const courses = { slalom: {}, spiral: {} };
  for (let difficulty = 0; difficulty < 3; difficulty++) {
    const count = 5 + difficulty;
    courses.slalom[difficulty + 1] = {
      name: `蛇形繞標・第${numerals[difficulty]}關：${ranks[difficulty]}`,
      gates: Array.from({ length: count }, (_, i) => ({
        pos: [(i % 2 ? -1 : 1) * (0.55 + difficulty * 0.05),
          1.3 + (difficulty === 2 ? (i % 2 ? 0.2 : -0.2) : 0),
          -1.35 + i * 3.15 / (count - 1)],
        normal: [0, 0, 1], r: 0.34 - difficulty * 0.035,
        shape: 'ring', color: i === count - 1 ? 0xfbbf24 : 0x38bdf8
      }))
    };
    const spiralCount = 6 + difficulty * 2;
    const turns = 1 + difficulty * 0.25;
    courses.spiral[difficulty + 1] = {
      name: `螺旋爬升・第${numerals[difficulty]}關：${ranks[difficulty]}`,
      gates: Array.from({ length: spiralCount }, (_, i) => {
        const angle = Math.PI + i * Math.PI * 2 * turns / (spiralCount - 1);
        return {
          pos: [0.72 * Math.sin(angle), 0.65 + i * 1.6 / (spiralCount - 1),
            0.35 + 0.95 * Math.cos(angle)],
          normal: [Math.cos(angle), 0, -Math.sin(angle)],
          r: 0.32 - difficulty * 0.03, shape: 'ring',
          color: i === spiralCount - 1 ? 0xfbbf24 : 0xa78bfa
        };
      })
    };
  }
  root.DroneRacingCourses = courses;
  if (typeof module !== 'undefined' && module.exports) module.exports = courses;
})(typeof window === 'undefined' ? globalThis : window);
