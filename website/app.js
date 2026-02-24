/**
 * app.js — Frontend for LLMDEX v3: Multi-leaderboard Benchmark Intelligence.
 *
 * SAFETY DESIGN:
 *   - NEVER invents missing values. All null/undefined fields display as "N/A".
 *   - Sentiment data is always labeled EXPERIMENTAL.
 *   - Three leaderboards are clearly separated.
 *   - Rankings are explainable — users understand why a model ranks where it does.
 *
 * FEATURES:
 *   - Three independent leaderboards: Performance, Value, Efficiency
 *   - Performance breakdown modal (Part 5)
 *   - Community sentiment panel (Part 6)
 *   - AI advisor engine (Part 7)
 *   - Historical trend explorer (Part 8)
 *   - Sortable columns, comparison mode, tooltips
 */

let currentTab = "performance-tab";
let allDataRef = null;
let currentSort = { field: null, direction: null };

document.addEventListener("DOMContentLoaded", async () => {
  const hamburgerBtn = document.getElementById("hamburger-btn");
  const navLinks = document.getElementById("nav-links");
  if (hamburgerBtn && navLinks) {
    hamburgerBtn.addEventListener("click", () => {
      navLinks.classList.toggle("show");
    });
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("show");
      });
    });
  }

  const DATA_URLS = ["../data/index/latest.json", "../data/models.json"];

  let allData = [];
  let dataLoaded = false;
  let columnDefs = {};
  let sentimentData = [];

  // ── Load column definitions ──
  try {
    const defRes = await fetch("../data/column_definitions.json");
    if (defRes.ok) columnDefs = await defRes.json();
  } catch (e) {
    console.warn("Column definitions not loaded:", e);
  }

  // ── Load main data ──
  for (const url of DATA_URLS) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        allData = await response.json();
        dataLoaded = true;
        break;
      }
    } catch (e) {
      console.warn(`Failed to load ${url}:`, e);
    }
  }

  if (!dataLoaded || allData.length === 0) {
    const banner = document.getElementById("banner-area");
    if (banner) {
      banner.innerHTML = `<div class="error-msg">
        <i class="fas fa-exclamation-triangle"></i>
        Data loading failed. Ensure the pipeline has been run successfully.
      </div>`;
    }
    return;
  }

  // ── Load sentiment data ──
  try {
    const sentRes = await fetch("../data/sentiment/latest.json");
    if (sentRes.ok) sentimentData = await sentRes.json();
  } catch (e) {
    console.warn("Sentiment data not loaded:", e);
  }

  // Family Growth Explorer reads from allData directly (no separate history fetch needed)

  // ── Remove skeleton states ──
  document
    .querySelectorAll(".skeleton")
    .forEach((el) => el.classList.remove("skeleton"));
  document
    .querySelectorAll(".skeleton-chart")
    .forEach((el) => el.classList.remove("skeleton-chart"));

  // ── Check dataset completeness ──
  const sources = new Set();
  allData.forEach((d) => {
    if (d.sources) {
      (typeof d.sources === "string"
        ? JSON.parse(d.sources)
        : d.sources
      ).forEach((s) => sources.add(s));
    }
  });

  if (sources.size < 2) {
    const banner = document.getElementById("banner-area");
    if (banner) {
      banner.innerHTML = `<div class="dataset-warning glass">
        <i class="fas fa-exclamation-circle"></i>
        <strong>Incomplete Dataset:</strong> Only ${sources.size} source(s) contributed data.
        Rankings may be less reliable than usual.
      </div>`;
    }
  }

  allDataRef = allData;
  renderDashboard(allData);
  setupLeaderboardTabs(allData);
  setupTooltips(columnDefs);
  setupSorting(allData);
  setupComparison(allData);
  setupBenchmarkModal(allData);
  renderSentimentPanel(sentimentData);
  setupFamilyExplorer(allData);
  setupAdvisor(allData);
  setupChatbot(allData);

  // ── Filters ──
  const searchInput = document.getElementById("search-input");
  const providerFilter = document.getElementById("provider-filter");

  const filterData = () => {
    const query = searchInput.value.toLowerCase();
    const provider = providerFilter.value;
    const filtered = filterModels(allData, query, provider);
    populateTable(filtered, allData, currentTab);
  };

  searchInput.addEventListener("input", filterData);
  providerFilter.addEventListener("change", filterData);
});

function filterModels(data, query, provider) {
  return data.filter((d) => {
    const name = (d.canonical_name || d.model_name || "").toLowerCase();
    const matchesSearch = name.includes(query);
    const matchesProvider =
      provider === "All" ||
      d.provider === provider ||
      (provider === "Other" &&
        ![
          "OpenAI",
          "Anthropic",
          "Google",
          "Meta",
          "DeepSeek",
          "Mistral AI",
          "Alibaba",
          "Zhipu AI",
          "xAI",
          "Moonshot",
          "MiniMax",
        ].includes(d.provider));
    return matchesSearch && matchesProvider;
  });
}

// ══════════════════════════════════════════════════════════════
// DASHBOARD RENDERER
// ══════════════════════════════════════════════════════════════

function renderDashboard(data) {
  if (!data || data.length === 0) return;

  const displayName = (m) => m.canonical_name || m.model_name || "Unknown";

  // ── Top Performer (by performance_rank) ──
  const perfRanked = data
    .filter((d) => d.performance_rank != null)
    .sort((a, b) => a.performance_rank - b.performance_rank);
  const topPerf = perfRanked[0];
  if (topPerf) {
    document.getElementById("top-model").textContent = displayName(topPerf);
    document.getElementById("top-score").textContent =
      topPerf.adjusted_performance != null
        ? `Adj. Performance: ${topPerf.adjusted_performance.toFixed(1)}`
        : topPerf.performance_index != null
          ? `Performance: ${topPerf.performance_index.toFixed(1)}`
          : "Performance: N/A";
  }

  // ── Best Value (by value_rank) ──
  const valueRanked = data
    .filter((d) => d.value_rank != null)
    .sort((a, b) => a.value_rank - b.value_rank);
  const topVal = valueRanked[0];
  if (topVal) {
    document.getElementById("val-model").textContent = displayName(topVal);
    document.getElementById("val-score").textContent =
      topVal.composite_index != null
        ? `Composite: ${topVal.composite_index.toFixed(1)}`
        : "N/A";
  }

  // ── Most Efficient ──
  const effRanked = data
    .filter((d) => d.efficiency_rank != null)
    .sort((a, b) => a.efficiency_rank - b.efficiency_rank);
  const topEff = effRanked[0];
  if (topEff) {
    document.getElementById("eff-model").textContent = displayName(topEff);
    document.getElementById("eff-score").textContent =
      topEff.efficiency_score != null
        ? `Efficiency: ${topEff.efficiency_score.toFixed(1)}th pctl`
        : "N/A";
  } else {
    document.getElementById("eff-model").textContent = "N/A";
    document.getElementById("eff-score").textContent = "N/A";
  }

  document.getElementById("total-models").textContent = data.length;

  const sourceCountEl = document.getElementById("source-count");
  if (sourceCountEl) {
    const allSources = new Set();
    data.forEach((d) => {
      if (d.sources) {
        (typeof d.sources === "string"
          ? JSON.parse(d.sources)
          : d.sources
        ).forEach((s) => allSources.add(s));
      }
    });
    sourceCountEl.textContent = `Across ${allSources.size} sources`;
  }

  renderScatterPlot(data);
  renderEfficiencyChart(data);
  setupRadarChart(data, topPerf || valueRanked[0]);
  populateTable(data, data, "performance-tab");
}

// ══════════════════════════════════════════════════════════════
// LEADERBOARD TABS (Part 1 + Part 9)
// ══════════════════════════════════════════════════════════════

const TAB_DESCRIPTIONS = {
  "performance-tab": `<strong>Performance Leaderboard:</strong> Pure intelligence ranking based on 
    bias-corrected benchmark scores. No cost or speed influence. Models are ranked 
    by adjusted_performance which accounts for data completeness (confidence factor).`,
  "value-tab": `<strong>Value Leaderboard:</strong> Composite ranking blending 50% Performance + 
    30% Cost Efficiency + 20% Speed. Best for choosing a well-rounded model. 
    Uses adjusted performance for fairness.`,
  "efficiency-tab": `<strong>Efficiency Leaderboard:</strong> Performance per dollar, using percentile 
    normalization. Only models with adjusted performance ≥ 60 are included. 
    Best for cost-sensitive deployments.`,
};

function setupLeaderboardTabs(data) {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const tabId = tab.dataset.tab;
      currentTab = tabId;

      // Reset column sort when switching tabs
      currentSort = { field: null, direction: null };

      // Update description
      document.getElementById("tab-description-text").innerHTML =
        TAB_DESCRIPTIONS[tabId];

      // Re-populate table
      const query = document.getElementById("search-input").value.toLowerCase();
      const provider = document.getElementById("provider-filter").value;
      const filtered = filterModels(data, query, provider);
      populateTable(filtered, data, tabId);
    });
  });
}

// ══════════════════════════════════════════════════════════════
// SCATTER PLOT
// ══════════════════════════════════════════════════════════════

function renderScatterPlot(data) {
  const plotData = data.filter(
    (d) => d.blended_cost_per_1m != null && d.performance_index != null,
  );

  if (plotData.length === 0) {
    document.getElementById("scatter-chart").innerHTML =
      '<p style="text-align:center;color:var(--text-secondary);padding:2rem">No models with both cost and performance data available.</p>';
    return;
  }

  const trace = {
    x: plotData.map((d) => d.blended_cost_per_1m),
    y: plotData.map((d) => d.adjusted_performance || d.performance_index),
    text: plotData.map((d) => {
      const name = d.canonical_name || d.model_name;
      const cf =
        d.confidence_factor != null
          ? ` (CF: ${(d.confidence_factor * 100).toFixed(0)}%)`
          : "";
      return `${name}${cf}`;
    }),
    mode: "markers",
    type: "scatter",
    marker: {
      size: plotData.map((d) => 8 + (d.confidence_factor || 0.5) * 10),
      color: plotData.map(
        (d) => d.adjusted_performance || d.performance_index || 0,
      ),
      colorscale: "Viridis",
      showscale: window.innerWidth > 768,
      colorbar: { title: "Adj. Perf" },
      line: { color: "white", width: 1 },
    },
  };

  const isMobile = window.innerWidth <= 768;
  const layout = {
    title: {
      text: isMobile
        ? "Perf. vs Cost Frontier"
        : "Performance vs Cost Frontier (bias-corrected)",
      font: { color: "#f8fafc", size: isMobile ? 11 : undefined },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: isMobile ? { l: 40, r: 25, t: 30, b: 40 } : undefined,
    xaxis: {
      title: isMobile ? "Cost ($)" : "Blended Cost per 1M Tokens ($)",
      type: "log",
      color: "#94a3b8",
      tickfont: { size: isMobile ? 10 : 12 },
    },
    yaxis: {
      title: isMobile ? "Adj. Perf" : "Adjusted Performance (0-100)",
      color: "#94a3b8",
      tickfont: { size: isMobile ? 10 : 12 },
    },
    hovermode: "closest",
  };

  Plotly.newPlot("scatter-chart", [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ══════════════════════════════════════════════════════════════
// EFFICIENCY CHART (Part 3 — Percentile Normalization)
// ══════════════════════════════════════════════════════════════

function renderEfficiencyChart(data) {
  const validEfficiency = data.filter(
    (d) => d.efficiency_score != null && d.efficiency_rank != null,
  );

  validEfficiency.sort((a, b) => a.efficiency_rank - b.efficiency_rank);
  const top10 = validEfficiency.slice(0, 10).reverse();

  if (top10.length === 0) {
    document.getElementById("bar-chart").innerHTML =
      '<p style="text-align:center;color:var(--text-secondary);padding:2rem">No models with valid efficiency data available.</p>';
    return;
  }

  const trace = {
    x: top10.map((d) => d.efficiency_score),
    y: top10.map((d) => d.canonical_name || d.model_name),
    type: "bar",
    orientation: "h",
    marker: {
      color: top10.map((d) => d.efficiency_score),
      colorscale: [
        [0, "#1e40af"],
        [0.5, "#3b82f6"],
        [1, "#10b981"],
      ],
    },
    text: top10.map((d) => `${d.efficiency_score.toFixed(1)} pctl`),
    textposition: "outside",
    textfont: { color: "#94a3b8", size: 11 },
  };

  const isMobile = window.innerWidth <= 768;
  const layout = {
    title: {
      text: `Top ${top10.length} Most Efficient${isMobile ? "" : " (Percentile Rank)"}`,
      font: { color: "#f8fafc", size: isMobile ? 13 : undefined },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: isMobile ? "Percentile" : "Efficiency Percentile (0-100)",
      color: "#94a3b8",
      range: [0, 110],
      tickfont: { size: isMobile ? 10 : 12 },
    },
    yaxis: {
      color: "#f8fafc",
      automargin: true,
      tickfont: { size: isMobile ? 8 : 12 },
    },
    margin: isMobile ? { l: 110, r: 25, t: 30, b: 35 } : { l: 160, r: 60 },
  };

  Plotly.newPlot("bar-chart", [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ══════════════════════════════════════════════════════════════
// RADAR CHART
// ══════════════════════════════════════════════════════════════

function setupRadarChart(data, defaultModel) {
  const selector = document.getElementById("radar-model-selector");
  if (!selector) return;

  const ranked = data
    .filter((d) => d.performance_rank != null)
    .sort((a, b) => a.performance_rank - b.performance_rank);

  selector.innerHTML = "";
  ranked.forEach((m, i) => {
    const opt = document.createElement("option");
    const name = m.canonical_name || m.model_name;
    opt.value = i.toString();
    opt.textContent = `#${m.performance_rank || "?"} ${name}`;
    selector.appendChild(opt);
  });

  const findModel = (val) => ranked[parseInt(val)];

  if (defaultModel) renderRadarChart(defaultModel);

  selector.addEventListener("change", () => {
    const model = findModel(selector.value);
    if (model) renderRadarChart(model);
  });
}

function renderRadarChart(model) {
  if (!model) return;

  const axes = [
    {
      name: "Intelligence",
      val: model.intelligence_score,
      available: model.intelligence_score != null,
    },
    {
      name: "Coding",
      val: model.coding_score,
      available: model.coding_score != null,
    },
    {
      name: "GPQA",
      val: model.gpqa,
      available: model.gpqa != null,
    },
    {
      name: "Cost Efficiency",
      val: model.cost_index,
      available: model.cost_index != null,
    },
    {
      name: "Speed",
      val: model.speed_index,
      available: model.speed_index != null,
    },
  ];

  const filteredR = [];
  const filteredTheta = [];
  axes.forEach((a) => {
    if (a.available) {
      filteredR.push(a.val);
      filteredTheta.push(a.name);
    }
  });

  if (filteredR.length === 0) {
    document.getElementById("radar-chart").innerHTML =
      '<p style="text-align:center;color:var(--text-secondary);padding:2rem">No data available for this model.</p>';
    return;
  }

  const rPlot = [...filteredR, filteredR[0]];
  const thetaPlot = [...filteredTheta, filteredTheta[0]];

  const naAxes = axes.filter((a) => !a.available).map((a) => a.name);
  const titleSuffix = naAxes.length > 0 ? `\n(${naAxes.join(", ")}: N/A)` : "";
  const cfLabel =
    model.confidence_factor != null
      ? ` | CF: ${(model.confidence_factor * 100).toFixed(0)}%`
      : "";

  const trace = {
    type: "scatterpolar",
    r: rPlot,
    theta: thetaPlot,
    fill: "toself",
    name: model.canonical_name || model.model_name,
    line: { color: "#3b82f6" },
    fillcolor: "rgba(59, 130, 246, 0.15)",
  };

  const isMobile = window.innerWidth <= 768;
  const layout = {
    title: {
      text: `${model.canonical_name || model.model_name}${cfLabel}${titleSuffix}`,
      font: { color: "#f8fafc", size: isMobile ? 11 : 13 },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    polar: {
      radialaxis: {
        visible: true,
        range: [0, 100],
        color: "#94a3b8",
        tickfont: { size: isMobile ? 8 : 12 },
      },
      bgcolor: "rgba(0,0,0,0)",
      angularaxis: {
        color: "#f8fafc",
        tickfont: { size: isMobile ? 8 : 12 },
      },
    },
    showlegend: false,
    margin: isMobile ? { t: 40, b: 15, l: 30, r: 30 } : { t: 60, b: 30 },
  };

  Plotly.newPlot("radar-chart", [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ══════════════════════════════════════════════════════════════
// TABLE RENDERING — Multi-leaderboard (Parts 1, 2, 5)
// ══════════════════════════════════════════════════════════════

function formatContext(val) {
  if (val == null) return '<span class="na-badge">N/A</span>';
  const num = Number(val);
  if (isNaN(num)) return '<span class="na-badge">N/A</span>';
  if (num >= 1_000_000)
    return `${(num / 1_000_000).toFixed(num % 1_000_000 === 0 ? 0 : 1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toLocaleString();
}

function coverageBadge(count) {
  const n = Number(count) || 0;
  let cls = "coverage-low";
  if (n >= 3) cls = "coverage-high";
  else if (n >= 2) cls = "coverage-mid";
  return `<span class="coverage-badge ${cls}">${n || "?"}</span>`;
}

function confidenceBadge(cf) {
  if (cf == null) return '<span class="na-badge">N/A</span>';
  const pct = (cf * 100).toFixed(0);
  let cls = "coverage-low";
  if (cf >= 0.8) cls = "coverage-high";
  else if (cf >= 0.5) cls = "coverage-mid";
  return `<span class="coverage-badge ${cls}">${pct}%</span>`;
}

const TABLE_CONFIGS = {
  "performance-tab": {
    rankField: "performance_rank",
    sortDefault: "adjusted_performance",
    columns: [
      { key: "rank", label: "Rank", sortable: false },
      { key: "checkbox", label: "", sortable: false },
      { key: "model_name", label: "Model", sortable: false },
      { key: "provider", label: "Provider", sortable: false },
      {
        key: "adjusted_performance",
        label: "Adj. Performance",
        sortable: true,
        info: "performance",
        format: "score",
      },
      {
        key: "performance_index",
        label: "Raw Performance",
        sortable: true,
        format: "score",
      },
      {
        key: "confidence_factor",
        label: "Confidence",
        sortable: true,
        format: "confidence",
      },
      {
        key: "intelligence_score",
        label: "Intelligence",
        sortable: true,
        format: "raw_score",
      },
      {
        key: "coding_score",
        label: "Coding",
        sortable: true,
        format: "raw_score",
      },
      {
        key: "gpqa",
        label: "GPQA",
        sortable: true,
        format: "raw_score",
      },
      {
        key: "arena_elo",
        label: "Arena ELO",
        sortable: true,
        format: "elo",
      },
      {
        key: "arena_votes",
        label: "Votes",
        sortable: true,
        format: "integer",
      },
    ],
  },
  "value-tab": {
    rankField: "value_rank",
    sortDefault: "composite_index",
    columns: [
      { key: "rank", label: "Rank", sortable: false },
      { key: "checkbox", label: "", sortable: false },
      { key: "model_name", label: "Model", sortable: false },
      { key: "provider", label: "Provider", sortable: false },
      {
        key: "composite_index",
        label: "Composite",
        sortable: true,
        info: "composite",
        format: "score",
      },
      {
        key: "adjusted_performance",
        label: "Adj. Perf",
        sortable: true,
        format: "score",
      },
      {
        key: "cost_index",
        label: "Cost Score",
        sortable: true,
        format: "score",
      },
      {
        key: "speed_index",
        label: "Speed Score",
        sortable: true,
        format: "score",
      },
      {
        key: "blended_cost_per_1m",
        label: "Cost ($/1M)",
        sortable: true,
        info: "cost_input",
        format: "cost",
      },
      {
        key: "tokens_per_second",
        label: "Speed (t/s)",
        sortable: true,
        format: "raw_score",
      },
      {
        key: "latency_seconds",
        label: "Latency (s)",
        sortable: true,
        format: "latency",
      },
      {
        key: "context_window",
        label: "Context",
        sortable: true,
        info: "context",
        format: "context",
      },
    ],
  },
  "efficiency-tab": {
    rankField: "efficiency_rank",
    sortDefault: "efficiency_score",
    columns: [
      { key: "rank", label: "Rank", sortable: false },
      { key: "checkbox", label: "", sortable: false },
      { key: "model_name", label: "Model", sortable: false },
      { key: "provider", label: "Provider", sortable: false },
      {
        key: "efficiency_score",
        label: "Efficiency (Pctl)",
        sortable: true,
        format: "score",
      },
      {
        key: "adjusted_performance",
        label: "Adj. Perf",
        sortable: true,
        format: "score",
      },
      {
        key: "blended_cost_per_1m",
        label: "Cost ($/1M)",
        sortable: true,
        format: "cost",
      },
      {
        key: "tokens_per_second",
        label: "Speed (t/s)",
        sortable: true,
        format: "raw_score",
      },
      {
        key: "latency_seconds",
        label: "Latency (s)",
        sortable: true,
        format: "latency",
      },
      {
        key: "confidence_factor",
        label: "Confidence",
        sortable: true,
        format: "confidence",
      },
      {
        key: "source_count",
        label: "Sources",
        sortable: true,
        format: "count",
      },
    ],
  },
};

function populateTable(data, fullData, tabId) {
  const config = TABLE_CONFIGS[tabId || "performance-tab"];
  const headerRow = document.getElementById("table-header-row");
  const tbody = document.querySelector("#models-table tbody");

  // Build header
  headerRow.innerHTML = "";
  config.columns.forEach((col) => {
    const th = document.createElement("th");
    if (col.key === "checkbox") {
      th.innerHTML =
        '<input type="checkbox" id="select-all-models" title="Select all for comparison" />';
    } else if (col.sortable) {
      th.dataset.sort = col.key;
      th.classList.add("sortable");
      th.innerHTML = `${col.label} ${col.info ? `<span class="info-icon" data-col="${col.info}"><i class="fas fa-info-circle"></i></span>` : ""} <span class="sort-indicator"></span>`;
    } else {
      th.textContent = col.label;
    }
    headerRow.appendChild(th);
  });

  // Re-bind select-all
  const selectAll = document.getElementById("select-all-models");
  if (selectAll) {
    selectAll.addEventListener("change", () => {
      document.querySelectorAll(".model-checkbox").forEach((cb) => {
        cb.checked = selectAll.checked;
      });
      updateCompareCount();
    });
  }

  // Sort and filter data for this tab
  let tableData = [...data];
  const rankField = config.rankField;

  // For efficiency tab, only show eligible models
  if (tabId === "efficiency-tab") {
    tableData = tableData.filter((d) => d.efficiency_rank != null);
  }

  // Only apply default rank sort if user hasn't clicked a column sort
  if (!currentSort.field) {
    tableData.sort((a, b) => {
      const ra = a[rankField],
        rb = b[rankField];
      if (ra == null && rb == null) return 0;
      if (ra == null) return 1;
      if (rb == null) return -1;
      return ra - rb;
    });
  }

  tbody.innerHTML = "";
  if (tableData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${config.columns.length}" style="text-align:center; padding: 2rem;">No models found matching criteria.</td></tr>`;
    return;
  }

  const bests = computeBests(fullData, config.columns);

  tableData.forEach((row) => {
    const tr = document.createElement("tr");
    const modelId =
      row.model_slug || row.canonical_name || row.model_name || "";
    tr.dataset.modelSlug = modelId;

    config.columns.forEach((col) => {
      const td = document.createElement("td");

      if (col.key === "rank") {
        const rank = row[rankField];
        td.textContent = rank != null ? `#${Math.round(rank)}` : "—";
      } else if (col.key === "checkbox") {
        td.innerHTML = `<input type="checkbox" class="model-checkbox" data-slug="${modelId}" />`;
      } else if (col.key === "model_name") {
        const name = row.canonical_name || row.model_name || "Unknown";
        const hasBreakdown =
          row.benchmark_breakdown &&
          typeof row.benchmark_breakdown === "object";
        const breakdown =
          typeof row.benchmark_breakdown === "string"
            ? JSON.parse(row.benchmark_breakdown)
            : row.benchmark_breakdown;
        const showExpand = breakdown && Object.keys(breakdown).length > 0;
        // Data source indicator
        const tier = row.data_tier || 3;
        const tierBadge = tier === 1 ? '<span class="tier-badge tier-1" title="Cross-referenced (AA + LMSYS)">★</span>' 
                        : tier === 3 ? '<span class="tier-badge tier-3" title="LMSYS Arena only">○</span>' 
                        : '';
        td.style.fontWeight = "600";
        td.style.color = "var(--text-primary)";
        td.innerHTML = `${tierBadge}${name} ${showExpand ? `<button class="perf-detail-btn" data-slug="${row.model_slug || ""}" title="View benchmark breakdown"><i class="fas fa-chart-bar"></i></button>` : ""}`;
      } else if (col.key === "provider") {
        td.innerHTML = row.provider || '<span class="na-badge">—</span>';
      } else {
        const val = row[col.key];
        if (val == null) {
          td.innerHTML = '<span class="na-badge">–</span>';
        } else if (col.format === "score") {
          td.textContent = Number(val).toFixed(1);
        } else if (col.format === "cost") {
          td.textContent = `$${Number(val).toFixed(2)}`;
        } else if (col.format === "context") {
          td.innerHTML = formatContext(val);
        } else if (col.format === "confidence") {
          td.innerHTML = confidenceBadge(val);
        } else if (col.format === "count") {
          td.innerHTML = coverageBadge(val);
        } else if (col.format === "raw_score") {
          td.textContent = typeof val === "number" ? val.toFixed(1) : val;
        } else if (col.format === "elo") {
          td.textContent =
            typeof val === "number" ? Math.round(val).toLocaleString() : val;
        } else if (col.format === "integer") {
          td.textContent =
            typeof val === "number" ? Math.round(val).toLocaleString() : val;
        } else if (col.format === "latency") {
          td.textContent = typeof val === "number" ? val.toFixed(2) + "s" : val;
        } else {
          td.textContent = val;
        }

        // Best-in-column highlight
        if (bests[col.key] !== undefined && val != null) {
          const isBest =
            col.format === "cost"
              ? val > 0 && val === bests[col.key]
              : val === bests[col.key];
          if (isBest) td.classList.add("best-in-col");
        }
      }
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  // Restore sort indicator if active sort matches a column in this tab
  if (currentSort.field) {
    const activeTh = headerRow.querySelector(
      `th[data-sort="${currentSort.field}"]`,
    );
    if (activeTh) {
      activeTh.classList.add(
        currentSort.direction === "asc" ? "sort-asc" : "sort-desc",
      );
    }
  }
}

function computeBests(data, columns) {
  const bests = {};
  columns.forEach((col) => {
    if (!col.sortable || col.key === "confidence_factor") return;
    const isCost = col.format === "cost";
    let best = null;
    data.forEach((d) => {
      const v = d[col.key];
      if (v == null) return;
      if (isCost) {
        if (v > 0 && (best === null || v < best)) best = v;
      } else {
        if (best === null || v > best) best = v;
      }
    });
    bests[col.key] = best;
  });
  return bests;
}

// ══════════════════════════════════════════════════════════════
// BENCHMARK BREAKDOWN MODAL (Part 5)
// ══════════════════════════════════════════════════════════════

function setupBenchmarkModal(data) {
  const modal = document.getElementById("benchmark-modal");
  const closeBtn = document.getElementById("modal-close");
  if (!modal || !closeBtn) return;

  closeBtn.addEventListener("click", () => {
    modal.style.display = "none";
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.style.display = "none";
  });

  // Delegate clicks on perf-detail-btn
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".perf-detail-btn");
    if (!btn) return;

    const slug = btn.dataset.slug;
    const model = data.find((d) => (d.model_slug || d.model_name) === slug);
    if (!model) return;

    showBenchmarkModal(model);
  });
}

function showBenchmarkModal(model) {
  const modal = document.getElementById("benchmark-modal");
  const modalName = document.getElementById("modal-model-name");
  const modalBody = document.getElementById("modal-body");

  const name = model.canonical_name || model.model_name || "Unknown";
  modalName.textContent = `${name} — Benchmark Breakdown`;

  let html = "";

  // Overall scores
  html += `<div class="modal-scores-grid">`;
  const scoreFields = [
    {
      key: "intelligence_score",
      label: "Intelligence Index (AA)",
      icon: "fa-brain",
    },
    { key: "coding_score", label: "Coding Index (AA)", icon: "fa-code" },
    { key: "arena_elo", label: "Arena Elo", icon: "fa-trophy" },
    { key: "arena_votes", label: "Arena Votes", icon: "fa-users" },
    { key: "gpqa", label: "GPQA Diamond", icon: "fa-flask" },
    { key: "aime25", label: "AIME 2025", icon: "fa-calculator" },
    { key: "hle", label: "Humanity's Last Exam", icon: "fa-graduation-cap" },
    { key: "livecodebench", label: "LiveCodeBench", icon: "fa-laptop-code" },
    { key: "gdpval", label: "GDPval-AA (Agentic)", icon: "fa-robot" },
    {
      key: "blended_cost_per_1m",
      label: "Blended Cost ($/1M)",
      icon: "fa-dollar-sign",
    },
    {
      key: "tokens_per_second",
      label: "Speed (t/s)",
      icon: "fa-tachometer-alt",
    },
    { key: "latency_seconds", label: "Latency (s)", icon: "fa-clock" },
  ];

  scoreFields.forEach((sf) => {
    const val = model[sf.key];
    const display =
      val != null ? (typeof val === "number" ? val.toFixed(1) : val) : "N/A";
    const cls = val != null ? "has-data" : "no-data";
    html += `<div class="modal-score-card ${cls}">
      <i class="fas ${sf.icon}"></i>
      <div class="score-label">${sf.label}</div>
      <div class="score-value">${display}</div>
    </div>`;
  });
  html += `</div>`;

  // Confidence factor
  const cf = model.confidence_factor;
  if (cf != null) {
    const pct = (cf * 100).toFixed(0);
    html += `<div class="modal-confidence">
      <div class="confidence-bar-container">
        <div class="confidence-bar" style="width:${pct}%"></div>
      </div>
      <span>Data Confidence: ${pct}% (${(model.sources || []).length}/2 sources)</span>
    </div>`;
  }

  // Adjusted vs raw performance
  if (model.adjusted_performance != null && model.performance_index != null) {
    const diff = model.adjusted_performance - model.performance_index;
    const sign = diff >= 0 ? "+" : "";
    html += `<div class="modal-adjustment">
      Raw Performance: <strong>${model.performance_index.toFixed(1)}</strong> → 
      Adjusted: <strong>${model.adjusted_performance.toFixed(1)}</strong>
      <span class="adj-diff">(${sign}${diff.toFixed(1)} bias correction)</span>
    </div>`;
  }

  // Per-source breakdown
  const breakdown =
    typeof model.benchmark_breakdown === "string"
      ? JSON.parse(model.benchmark_breakdown)
      : model.benchmark_breakdown;

  if (breakdown && Object.keys(breakdown).length > 0) {
    html += `<h4 class="modal-sub-header">Per-Source Breakdown</h4>`;
    html += '<div class="breakdown-grid">';
    const labelMap = {
      intelligence_score: "Intelligence",
      coding_score: "Coding",
      reasoning_score: "Reasoning",
      multimodal_score: "Multimodal",
      arena_elo: "Arena Elo",
    };
    for (const [source, metrics] of Object.entries(breakdown)) {
      html += `<div class="breakdown-source">
        <h4>${source}</h4>`;
      for (const [key, val] of Object.entries(metrics)) {
        const label = labelMap[key] || key.replace(/_/g, " ");
        html += `<div class="breakdown-metric">
          <span>${label}</span>
          <span class="bm-val">${typeof val === "number" ? val.toFixed(2) : val}</span>
        </div>`;
      }
      html += "</div>";
    }
    html += "</div>";
  }

  // Rankings summary
  html += `<div class="modal-ranks">
    <div class="rank-chip perf">Performance: ${model.performance_rank != null ? `#${model.performance_rank}` : "N/A"}</div>
    <div class="rank-chip value">Value: ${model.value_rank != null ? `#${model.value_rank}` : "N/A"}</div>
    <div class="rank-chip eff">Efficiency: ${model.efficiency_rank != null ? `#${model.efficiency_rank}` : "N/A"}</div>
  </div>`;

  modalBody.innerHTML = html;
  modal.style.display = "flex";
}

// ══════════════════════════════════════════════════════════════
// COMMUNITY SENTIMENT PANEL (Part 6) — Redesigned
// ══════════════════════════════════════════════════════════════

/**
 * Map numeric sentiment to human-readable label + CSS class.
 */
function getSentimentLabel(score) {
  if (score == null) return { label: "N/A", cls: "neutral", color: "#94a3b8" };
  if (score >= 0.25)
    return {
      label: "Strongly Positive",
      cls: "strongly-positive",
      color: "#10b981",
    };
  if (score >= 0.1)
    return { label: "Positive", cls: "positive", color: "#6ee7a0" };
  if (score > -0.1)
    return { label: "Mixed / Neutral", cls: "neutral", color: "#94a3b8" };
  if (score > -0.25)
    return {
      label: "Mostly Negative",
      cls: "mostly-negative",
      color: "#f59e0b",
    };
  return {
    label: "Strongly Negative",
    cls: "strongly-negative",
    color: "#ef4444",
  };
}

/**
 * Render 3 sentiment charts using Plotly.
 */
function renderSentimentCharts(sentimentData) {
  if (!sentimentData || sentimentData.length === 0) return;

  const isMobile = window.innerWidth <= 768;
  const CHART_HEIGHT = isMobile ? 300 : 450;
  const LEFT_MARGIN = isMobile ? 80 : 260;
  const RIGHT_MARGIN = isMobile ? 35 : 110;
  const MAX_MODELS = 10;
  const MAX_LABEL = isMobile ? 8 : 45;

  const clipName = (name) =>
    name.length > MAX_LABEL ? name.slice(0, MAX_LABEL - 3) + "..." : name;

  const baseLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 8, b: isMobile ? 40 : 50, l: LEFT_MARGIN, r: RIGHT_MARGIN },
    font: {
      color: "#e2e8f0",
      size: isMobile ? 11 : 13,
      family: "Inter, system-ui, sans-serif",
    },
    xaxis: {
      color: "#94a3b8",
      gridcolor: "rgba(255,255,255,0.06)",
      zeroline: false,
      tickfont: { size: isMobile ? 9 : 12 },
    },
    yaxis: {
      color: "#e2e8f0",
      gridcolor: "rgba(255,255,255,0.03)",
      autorange: "reversed",
      type: "category",
      tickfont: { size: isMobile ? 9 : 13, color: "#e2e8f0" },
    },
    height: CHART_HEIGHT,
    bargap: 0.1,
  };
  const plotlyConfig = { responsive: true, displayModeBar: false };

  // Rich tooltip builder
  const buildHoverText = (s) => {
    const sent =
      s.community_sentiment != null ? s.community_sentiment.toFixed(3) : "N/A";
    const mentions = s.mention_count || 0;
    const controversy =
      s.controversy_index != null
        ? (s.controversy_index * 100).toFixed(0) + "%"
        : "N/A";
    return `<b>${s.model_name}</b><br>Sentiment: ${sent}<br>Mentions: ${mentions}<br>Controversy: ${controversy}`;
  };

  // ──── 1) Most Positively Received ────
  const liked = [...sentimentData]
    .filter((s) => s.community_sentiment != null)
    .sort((a, b) => b.community_sentiment - a.community_sentiment)
    .slice(0, MAX_MODELS);

  const likedColors = liked.map((s) => {
    const v = s.community_sentiment;
    if (v > 0.25) return "#10b981";
    if (v > 0.1) return "#6ee7a0";
    if (v > 0) return "#94a3b8";
    return "#f59e0b";
  });

  Plotly.newPlot(
    "sentiment-liked-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: liked.map((s) => clipName(s.model_name)),
        x: liked.map((s) => s.community_sentiment),
        marker: { color: likedColors, line: { width: 0 }, cornerradius: 4 },
        width: 0.75, // Force bar thickness
        text: liked.map((s) => " " + s.community_sentiment.toFixed(3)),
        textposition: "outside",
        textfont: { color: "#e2e8f0", size: isMobile ? 9 : 12 },
        hovertext: liked.map(buildHoverText),
        hoverinfo: "text",
        cliponaxis: false,
      },
    ],
    {
      ...baseLayout,
      xaxis: {
        ...baseLayout.xaxis,
        title: {
          text: "Sentiment Score",
          font: { color: "#94a3b8", size: 11 },
        },
        range: [
          Math.min(0, ...liked.map((s) => s.community_sentiment)) - 0.05,
          Math.max(...liked.map((s) => s.community_sentiment)) * 1.45,
        ],
      },
    },
    plotlyConfig,
  );

  // ──── 2) Most Controversial ────
  const controversial = [...sentimentData]
    .filter((s) => s.controversy_index != null && s.controversy_index > 0)
    .sort((a, b) => b.controversy_index - a.controversy_index)
    .slice(0, MAX_MODELS);

  if (controversial.length > 0) {
    Plotly.newPlot(
      "sentiment-controversy-chart",
      [
        {
          type: "bar",
          orientation: "h",
          y: controversial.map((s) => clipName(s.model_name)),
          x: controversial.map((s) => s.controversy_index * 100),
          marker: {
            color: controversial.map((_, i) => {
              const ratio = 1 - i / Math.max(1, controversial.length - 1);
              return `rgba(245, 158, 11, ${0.45 + ratio * 0.55})`;
            }),
            line: { width: 0 },
            cornerradius: 4,
          },
          width: 0.75,
          text: controversial.map(
            (s) => " " + (s.controversy_index * 100).toFixed(1) + "%",
          ),
          textposition: "outside",
          textfont: { color: "#e2e8f0", size: isMobile ? 9 : 12 },
          hovertext: controversial.map(buildHoverText),
          hoverinfo: "text",
          cliponaxis: false,
        },
      ],
      {
        ...baseLayout,
        xaxis: {
          ...baseLayout.xaxis,
          title: {
            text: "Controversy Index %",
            font: { color: "#94a3b8", size: 11 },
          },
          range: [
            0,
            Math.max(...controversial.map((s) => s.controversy_index * 100)) *
              1.6,
          ],
        },
      },
      plotlyConfig,
    );
  } else {
    document.getElementById("sentiment-controversy-chart").innerHTML =
      `<p style="text-align:center;color:var(--text-secondary);padding:2rem">No controversial models detected.</p>`;
  }

  // ──── 3) Most Discussed ────
  const trending = [...sentimentData]
    .filter((s) => s.mention_count > 0)
    .sort((a, b) => b.mention_count - a.mention_count)
    .slice(0, MAX_MODELS);

  // Use rank-based distinct colors so identical counts still look different
  const discussedPalette = [
    "#818cf8", // indigo
    "#6366f1",
    "#8b5cf6", // violet
    "#a78bfa",
    "#c084fc", // purple
    "#7c3aed",
    "#6d28d9",
    "#5b21b6",
  ];

  Plotly.newPlot(
    "sentiment-trending-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: trending.map((s) => clipName(s.model_name)),
        x: trending.map((s) => s.mention_count),
        marker: {
          color: trending.map(
            (_, i) => discussedPalette[i % discussedPalette.length],
          ),
          line: { width: 0 },
          cornerradius: 4,
        },
        width: 0.75,
        text: trending.map((s) => " " + String(s.mention_count) + " mentions"),
        textposition: "outside",
        textfont: { color: "#e2e8f0", size: isMobile ? 9 : 12 },
        hovertext: trending.map(buildHoverText),
        hoverinfo: "text",
        cliponaxis: false,
      },
    ],
    {
      ...baseLayout,
      xaxis: {
        ...baseLayout.xaxis,
        title: { text: "Total Mentions", font: { color: "#94a3b8", size: 11 } },
        range: [0, Math.max(...trending.map((s) => s.mention_count)) * 1.5],
      },
    },
    plotlyConfig,
  );
}

/**
 * Render the sentiment cards grid with human labels & quotes.
 *
 * Cards are DYNAMICALLY selected as the union of:
 *   - Top 8 most positively received  (by community_sentiment)
 *   - Top 8 most controversial        (by controversy_index)
 *   - Top 8 most discussed            (by mention_count)
 * …deduplicated by model_name, then sorted by mention_count desc.
 * This means cards always mirror exactly what the charts show — no hardcoding.
 */
function renderSentimentPanel(sentimentData) {
  const grid = document.getElementById("sentiment-grid");
  if (!grid) return;

  if (!sentimentData || sentimentData.length === 0) {
    grid.innerHTML = `<div class="sentiment-empty glass">
      <i class="fas fa-satellite-dish"></i>
      <p>No sentiment data available yet. Run the sentiment pipeline to collect community data.</p>
      <code>python pipeline/sentiment_pipeline.py</code>
    </div>`;
    return;
  }

  // Render 3 charts above the cards
  renderSentimentCharts(sentimentData);

  // ── Dynamically build the card set from chart data ──
  const MAX_CHART = 10;

  const liked = [...sentimentData]
    .filter((s) => s.community_sentiment != null)
    .sort((a, b) => b.community_sentiment - a.community_sentiment)
    .slice(0, MAX_CHART);

  const controversial = [...sentimentData]
    .filter((s) => s.controversy_index != null && s.controversy_index > 0)
    .sort((a, b) => b.controversy_index - a.controversy_index)
    .slice(0, MAX_CHART);

  const trending = [...sentimentData]
    .filter((s) => s.mention_count > 0)
    .sort((a, b) => b.mention_count - a.mention_count)
    .slice(0, MAX_CHART);

  // Union — deduplicate by model_name, preserve full object
  const seen = new Set();
  const cardModels = [];
  for (const s of [...liked, ...controversial, ...trending]) {
    const key = s.model_name;
    if (!seen.has(key)) {
      seen.add(key);
      cardModels.push(s);
    }
  }

  // Sort cards by mention_count descending (most discussed first)
  cardModels.sort((a, b) => (b.mention_count || 0) - (a.mention_count || 0));

  // ── Render cards ──
  let html = "";
  cardModels.forEach((s) => {
    const score = s.community_sentiment;
    const { label, cls } = getSentimentLabel(score);
    const scoreDisplay = score != null ? score.toFixed(3) : "N/A";
    const trend = s.sentiment_trend || "N/A";
    const mentions = s.mention_count || 0;
    const controversy =
      s.controversy_index != null
        ? (s.controversy_index * 100).toFixed(1) + "%"
        : "N/A";

    let trendIcon = "fa-minus";
    if (trend === "positive") trendIcon = "fa-arrow-up";
    else if (trend === "negative") trendIcon = "fa-arrow-down";

    // Build quote HTML from real scraped data — supports both new and legacy keys
    const quotes = (s.community_examples || s.top_quotes || []).slice(0, 3);
    let quotesHtml = "";
    if (quotes.length > 0) {
      quotesHtml = `<div class="sentiment-quotes">
        <span class="quotes-label"><i class="fas fa-comments" style="margin-right:0.4rem;font-size:0.7rem;"></i>Recent community commentary</span>`;
      quotes.forEach((q) => {
        // Only render link if URL exists AND is a real link (not a placeholder/example)
        const isRealUrl =
          q.url && q.url.startsWith("http") && !/example/i.test(q.url);
        const linkIcon = isRealUrl
          ? `<a href="${q.url}" target="_blank" rel="noopener noreferrer" class="quote-link" title="View source"><i class="fas fa-external-link-alt"></i></a>`
          : "";
        quotesHtml += `<blockquote class="sentiment-quote">
          <span class="quote-text">"${q.text}"</span>
          <div class="quote-attribution">
            <cite>— ${q.source}</cite>
            ${linkIcon}
          </div>
        </blockquote>`;
      });
      quotesHtml += `</div>`;
    } else {
      quotesHtml = `<div class="sentiment-quotes">
        <p class="no-quotes">No high-signal community commentary found yet</p>
      </div>`;
    }

    // Confidence badge
    const confidence = s.confidence_score != null ? s.confidence_score : 0;
    let confLabel, confClass;
    if (confidence >= 0.7) {
      confLabel = "High";
      confClass = "confidence-high";
    } else if (confidence >= 0.35) {
      confLabel = "Medium";
      confClass = "confidence-medium";
    } else {
      confLabel = "Low";
      confClass = "confidence-low";
    }

    // Method badge
    const method = s.sentiment_method || "unknown";
    const methodLabel =
      method === "gemini"
        ? "Gemini"
        : method === "vader_fallback"
          ? "VADER"
          : method;
    const methodClass = method === "gemini" ? "method-gemini" : "method-vader";

    // Hover tooltip info
    const sources = ["Reddit", "Hacker News", "GitHub"].join(", ");
    const timeWindow = "30 days";
    const sampleSize = mentions;

    html += `<div class="sentiment-card glass" style="position:relative;">
      <div class="sentiment-header">
        <h4>${s.model_name || "Unknown"}</h4>
        <div style="display:flex;gap:0.4rem;align-items:center;flex-wrap:wrap;">
          <span class="confidence-badge ${confClass}" title="Confidence: ${confLabel}">
            <i class="fas fa-shield-alt" style="font-size:0.6rem;"></i> ${confLabel}
          </span>
          <span class="sentiment-method-badge ${methodClass}" title="Classified by ${methodLabel}">
            ${methodLabel}
          </span>
          <span class="sentiment-badge ${cls}" title="Raw score: ${scoreDisplay}">
            ${label}
          </span>
        </div>
      </div>
      <div class="sentiment-info-tooltip">
        <strong>Data Sources:</strong> ${sources}<br>
        <strong>Time Window:</strong> Last ${timeWindow}<br>
        <strong>Sample Size:</strong> ${sampleSize} posts<br>
        <strong>Confidence:</strong> ${confLabel} (${(confidence * 100).toFixed(0)}%)<br>
        <strong>Method:</strong> ${methodLabel}
      </div>
      <div class="sentiment-metrics">
        <div class="sent-metric">
          <i class="fas fa-comment"></i>
          <span>${mentions.toLocaleString()} mentions</span>
        </div>
        <div class="sent-metric">
          <i class="fas ${trendIcon}"></i>
          <span>Trend: ${trend}</span>
        </div>
        <div class="sent-metric">
          <i class="fas fa-exclamation-circle"></i>
          <span>Controversy: ${controversy}</span>
        </div>
      </div>
      ${quotesHtml}
    </div>`;
  });

  grid.innerHTML = html;

  // Add disclaimer below grid
  const disclaimerEl = document.createElement("div");
  disclaimerEl.className = "sentiment-disclaimer";
  disclaimerEl.innerHTML =
    '<i class="fas fa-flask"></i> Sentiment is AI-classified and experimental. It does not influence model rankings or scores.';
  grid.parentNode.appendChild(disclaimerEl);
}

// ══════════════════════════════════════════════════════════════
// AI ADVISOR (Part 7)
// ══════════════════════════════════════════════════════════════

function setupAdvisor(data) {
  allDataRef = data;
  const chips = document.querySelectorAll(".priority-chip");
  const selectedDiv = document.getElementById("selected-priorities");
  const goBtn = document.getElementById("advisor-go-btn");
  const resultsDiv = document.getElementById("advisor-results");
  let selectedPriorities = [];

  const updateUI = () => {
    // Update selected display
    selectedDiv.innerHTML =
      selectedPriorities.length === 0
        ? '<span class="no-selection">Click priorities above to add them</span>'
        : selectedPriorities
            .map(
              (p, i) =>
                `<span class="selected-chip">${i + 1}. ${p} <button class="remove-priority" data-idx="${i}">×</button></span>`,
            )
            .join("");

    goBtn.disabled = selectedPriorities.length === 0;

    // Update chip states
    chips.forEach((chip) => {
      const p = chip.dataset.priority;
      chip.classList.toggle("active", selectedPriorities.includes(p));
    });

    // Bind remove buttons
    selectedDiv.querySelectorAll(".remove-priority").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.idx);
        selectedPriorities.splice(idx, 1);
        updateUI();
      });
    });
  };

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const p = chip.dataset.priority;
      if (selectedPriorities.includes(p)) {
        selectedPriorities = selectedPriorities.filter((x) => x !== p);
      } else {
        selectedPriorities.push(p);
      }
      updateUI();
    });
  });

  goBtn.addEventListener("click", () => {
    const recommendations = getRecommendations(data, selectedPriorities);
    renderAdvisorResults(recommendations);
    resultsDiv.style.display = "block";
  });

  updateUI();
}

function getRecommendations(data, priorities, maxResults = 5) {
  const PRIORITY_CONFIG = {
    performance: {
      field: "adjusted_performance",
      label: "Performance",
      higherBetter: true,
    },
    cost: { field: "blended_cost_per_1m", label: "Cost", higherBetter: false },
    coding: { field: "coding_score", label: "Coding", higherBetter: true },
    reasoning: {
      field: "reasoning_score",
      label: "Reasoning",
      higherBetter: true,
    },
    efficiency: {
      field: "efficiency_score",
      label: "Efficiency",
      higherBetter: true,
    },
    context: {
      field: "context_window",
      label: "Context Window",
      higherBetter: true,
    },
  };

  const ranges = {
    adjusted_performance: [0, 100],
    performance_index: [0, 100],
    blended_cost_per_1m: [0, 100],
    coding_score: [0, 100],
    reasoning_score: [0, 100],
    efficiency_score: [0, 100],
    context_window: [0, 2000000],
  };

  // Compute weights
  const n = priorities.length;
  const weights = {};
  priorities.forEach((p, i) => {
    weights[p] = (n - i) / ((n * (n + 1)) / 2);
  });

  // Score candidates
  const scored = data
    .filter((d) => {
      const primaryField = PRIORITY_CONFIG[priorities[0]]?.field;
      return primaryField && d[primaryField] != null;
    })
    .map((d) => {
      let totalScore = 0;
      const details = {};

      for (const [priority, weight] of Object.entries(weights)) {
        const config = PRIORITY_CONFIG[priority];
        const val = d[config.field];
        if (val == null) continue;

        const [lo, hi] = ranges[config.field] || [0, 100];
        let norm = (val - lo) / (hi - lo);
        norm = Math.max(0, Math.min(1, norm));
        if (!config.higherBetter) norm = 1 - norm;

        totalScore += norm * weight;
        details[priority] = {
          raw: val,
          norm: norm.toFixed(2),
          weight: weight.toFixed(2),
        };
      }

      return { model: d, totalScore, details };
    })
    .sort((a, b) => b.totalScore - a.totalScore)
    .slice(0, maxResults);

  return scored.map((s, i) => {
    const m = s.model;
    const name = m.canonical_name || m.model_name || "Unknown";
    const provider = m.provider || "Unknown";

    // Build explanation
    const rankParts = [];
    if (m.performance_rank)
      rankParts.push(`#${m.performance_rank} in Performance`);
    if (m.efficiency_rank)
      rankParts.push(`#${m.efficiency_rank} in Efficiency`);
    if (m.value_rank) rankParts.push(`#${m.value_rank} in Value`);

    const useCases = {
      "coding,performance": "software engineering workflows",
      coding: "code generation and debugging",
      "reasoning,performance": "analytical and research workloads",
      reasoning: "mathematical problem solving",
      "efficiency,cost": "cost-sensitive production",
      efficiency: "balanced performance-to-cost",
      cost: "budget-conscious implementations",
      context: "long-document processing",
      performance: "high-capability general-purpose tasks",
    };

    const topTwo = priorities.slice(0, 2).join(",");
    const useCase =
      useCases[topTwo] || useCases[priorities[0]] || "general use";

    let explanation = `${name} by ${provider}. `;
    if (rankParts.length) explanation += rankParts.join(" and ") + ". ";
    explanation += `Recommended for ${useCase}.`;

    return {
      rank: i + 1,
      model_name: name,
      provider,
      totalScore: s.totalScore.toFixed(3),
      explanation,
      details: s.details,
      model_data: m,
    };
  });
}

function renderAdvisorResults(recommendations) {
  const div = document.getElementById("advisor-results");
  if (!recommendations || recommendations.length === 0) {
    div.innerHTML = '<p class="no-results">No models match your criteria.</p>';
    return;
  }

  let html =
    '<h4 class="results-header"><i class="fas fa-trophy"></i> Recommended Models</h4>';

  recommendations.forEach((rec) => {
    const m = rec.model_data;
    html += `<div class="advisor-result-card glass">
      <div class="result-rank">#${rec.rank}</div>
      <div class="result-info">
        <h4>${rec.model_name} <span class="result-provider">${rec.provider}</span></h4>
        <p class="result-explanation">${rec.explanation}</p>
        <div class="result-metrics">
          ${m.adjusted_performance != null ? `<span class="result-metric">Perf: ${m.adjusted_performance.toFixed(1)}</span>` : ""}
          ${m.blended_cost_per_1m != null ? `<span class="result-metric">Cost: $${m.blended_cost_per_1m.toFixed(2)}/1M</span>` : ""}
          ${m.efficiency_score != null ? `<span class="result-metric">Eff: ${m.efficiency_score.toFixed(1)} pctl</span>` : ""}
          ${m.coding_score != null ? `<span class="result-metric">Code: ${m.coding_score.toFixed(1)}</span>` : ""}
          ${m.reasoning_score != null ? `<span class="result-metric">Reason: ${m.reasoning_score.toFixed(1)}</span>` : ""}
        </div>
      </div>
      <div class="result-score">${rec.totalScore}</div>
    </div>`;
  });

  div.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════
// GEMINI-POWERED CHATBOT (Data-Grounded AI Advisor)
// ══════════════════════════════════════════════════════════════

function setupChatbot(data) {
  const messagesDiv = document.getElementById("chatbot-messages");
  const input = document.getElementById("chatbot-input");
  const sendBtn = document.getElementById("chatbot-send-btn");
  const clearBtn = document.getElementById("chatbot-clear-btn");

  if (!messagesDiv || !input || !sendBtn) return;

  // Rate limiting: max 5 per minute
  const requestTimestamps = [];
  const RATE_LIMIT = 5;
  const RATE_WINDOW = 60000; // 1 minute in ms

  // Response cache (10 min)
  const responseCache = new Map();
  const CACHE_TTL = 600000; // 10 minutes

  // Build compact dataset snapshot for client-side use
  const snapshot = buildDataSnapshot(data);

  function buildDataSnapshot(dataset) {
    const ranked = [...dataset]
      .filter((d) => d.performance_rank != null)
      .sort((a, b) => a.performance_rank - b.performance_rank)
      .slice(0, 15);

    return ranked.map((m) => ({
      model_name: m.canonical_name || m.model_name || "Unknown",
      provider: m.provider || "Unknown",
      performance_rank: m.performance_rank,
      value_rank: m.value_rank,
      efficiency_rank: m.efficiency_rank,
      adjusted_performance:
        m.adjusted_performance != null
          ? +m.adjusted_performance.toFixed(2)
          : null,
      blended_cost_per_1m:
        m.blended_cost_per_1m != null ? +m.blended_cost_per_1m.toFixed(2) : null,
      output_cost_per_1m:
        m.output_cost_per_1m != null ? +m.output_cost_per_1m.toFixed(2) : null,
      context_window: m.context_window,
      coding_score: m.coding_score != null ? +m.coding_score.toFixed(2) : null,
      reasoning_score:
        m.reasoning_score != null ? +m.reasoning_score.toFixed(2) : null,
      confidence_factor:
        m.confidence_factor != null ? +m.confidence_factor.toFixed(2) : null,
    }));
  }

  // Enable/disable send button based on input
  input.addEventListener("input", () => {
    sendBtn.disabled = input.value.trim().length === 0;
  });

  // Enter key to send
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && input.value.trim()) {
      e.preventDefault();
      handleSend();
    }
  });

  sendBtn.addEventListener("click", handleSend);

  // Clear conversation
  clearBtn.addEventListener("click", () => {
    messagesDiv.innerHTML = `<div class="chat-message bot-message">
      <div class="chat-avatar"><i class="fas fa-robot"></i></div>
      <div class="chat-bubble">
        <p>Conversation cleared. How can I help you with model analysis?</p>
        <div class="chat-suggestions">
          <button class="suggestion-chip" data-query="Which model has the best performance-to-cost ratio?">Best value?</button>
          <button class="suggestion-chip" data-query="Compare the top 3 models for coding tasks">Top coding models?</button>
          <button class="suggestion-chip" data-query="What is the cheapest high-performance model?">Cheapest good model?</button>
        </div>
      </div>
    </div>`;
    bindSuggestionChips();
  });

  // Bind suggestion chip clicks
  function bindSuggestionChips() {
    messagesDiv.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.dataset.query;
        if (query) {
          input.value = query;
          sendBtn.disabled = false;
          handleSend();
        }
      });
    });
  }
  bindSuggestionChips();

  function checkRateLimit() {
    const now = Date.now();
    // Remove old timestamps
    while (
      requestTimestamps.length &&
      now - requestTimestamps[0] > RATE_WINDOW
    ) {
      requestTimestamps.shift();
    }
    if (requestTimestamps.length >= RATE_LIMIT) {
      return false;
    }
    requestTimestamps.push(now);
    return true;
  }

  function checkCache(query) {
    const key = query.trim().toLowerCase();
    if (responseCache.has(key)) {
      const entry = responseCache.get(key);
      if (Date.now() - entry.timestamp < CACHE_TTL) {
        return entry.response;
      }
      responseCache.delete(key);
    }
    return null;
  }

  function storeCache(query, response) {
    const key = query.trim().toLowerCase();
    responseCache.set(key, { timestamp: Date.now(), response });
    // Limit cache size
    if (responseCache.size > 50) {
      const oldest = responseCache.keys().next().value;
      responseCache.delete(oldest);
    }
  }

  function addUserMessage(text) {
    const msgEl = document.createElement("div");
    msgEl.className = "chat-message user-message";
    msgEl.innerHTML = `
      <div class="chat-avatar"><i class="fas fa-user"></i></div>
      <div class="chat-bubble"><p>${escapeHtml(text)}</p></div>
    `;
    messagesDiv.appendChild(msgEl);
    scrollToBottom();
  }

  function addBotMessage(answer, referencedModels, dataPointsUsed, source) {
    const msgEl = document.createElement("div");
    msgEl.className = "chat-message bot-message";

    let refsHtml = "";
    if (referencedModels && referencedModels.length > 0) {
      refsHtml = `<div class="chat-refs">${referencedModels.map((m) => `<span class="chat-ref-tag">${escapeHtml(m)}</span>`).join("")}</div>`;
    }

    let sourceHtml = "";
    if (source === "cache")
      sourceHtml =
        '<div class="chat-source"><i class="fas fa-bolt"></i> Cached response</div>';
    else if (source === "client")
      sourceHtml =
        '<div class="chat-source"><i class="fas fa-database"></i> Analyzed from local data</div>';
    else if (source === "gemini")
      sourceHtml =
        '<div class="chat-source"><i class="fas fa-sparkles"></i> Powered by Gemini</div>';
    else if (source === "fallback")
      sourceHtml =
        '<div class="chat-source"><i class="fas fa-exclamation-triangle"></i> Fallback mode</div>';

    msgEl.innerHTML = `
      <div class="chat-avatar"><i class="fas fa-robot"></i></div>
      <div class="chat-bubble">
        <p>${formatAnswer(answer)}</p>
        ${refsHtml}
        ${sourceHtml}
      </div>
    `;
    messagesDiv.appendChild(msgEl);
    scrollToBottom();
  }

  function addTypingIndicator() {
    const typingEl = document.createElement("div");
    typingEl.className = "chat-message bot-message";
    typingEl.id = "chatbot-typing";
    typingEl.innerHTML = `
      <div class="chat-avatar"><i class="fas fa-robot"></i></div>
      <div class="chat-bubble">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
      </div>
    `;
    messagesDiv.appendChild(typingEl);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById("chatbot-typing");
    if (el) el.remove();
  }

  function scrollToBottom() {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function formatAnswer(text) {
    // Convert **bold** to <strong>
    let formatted = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Convert line breaks
    formatted = formatted.replace(/\n/g, "<br>");
    return formatted;
  }

  async function handleSend() {
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    sendBtn.disabled = true;

    // Add user message
    addUserMessage(query);

    // Rate limit check
    if (!checkRateLimit()) {
      addBotMessage(
        "You've reached the rate limit (5 queries per minute). Please wait a moment before asking again.",
        [],
        [],
        "rate_limit",
      );
      return;
    }

    // Check cache
    const cached = checkCache(query);
    if (cached) {
      addBotMessage(
        cached.answer,
        cached.referenced_models,
        cached.data_points_used,
        "cache",
      );
      return;
    }

    // Show typing indicator
    addTypingIndicator();

    try {
      // Try server-side API first, fall back to client-side analysis
      const response = await getAdvisorResponse(query);
      removeTypingIndicator();

      if (response) {
        addBotMessage(
          response.answer,
          response.referenced_models || [],
          response.data_points_used || [],
          response.source || "gemini",
        );
        storeCache(query, response);
      } else {
        // Client-side fallback
        const fallback = generateClientSideResponse(query, snapshot);
        addBotMessage(
          fallback.answer,
          fallback.referenced_models,
          fallback.data_points_used,
          "client",
        );
        storeCache(query, fallback);
      }
    } catch (error) {
      removeTypingIndicator();
      addBotMessage(
        "AI advisor temporarily unavailable. Please use the ranking filters and priority selector below to find the best models for your needs.",
        [],
        [],
        "fallback",
      );
    }
  }

  async function getAdvisorResponse(query) {
    // Try the local API endpoint
    try {
      const resp = await fetch("/api/advisor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query }),
      });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      // Server not available, use client-side fallback
    }
    return null;
  }

  /**
   * Client-side intelligent response when Gemini API is unavailable.
   * Uses the local dataset snapshot to answer common queries analytically.
   */
  function generateClientSideResponse(query, snap) {
    const q = query.toLowerCase();
    const models = snap.filter((m) => m.adjusted_performance != null);

    // Best performance
    if (q.includes("best") && (q.includes("perform") || q.includes("top"))) {
      const sorted = [...models]
        .sort((a, b) => a.performance_rank - b.performance_rank)
        .slice(0, 5);
      const answer =
        `Based on LLMDEX data, the top performing models are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — Performance: ${m.adjusted_performance}, Rank #${m.performance_rank}`,
          )
          .join("\n") +
        `\n\nThese rankings are based on adjusted performance scores from multiple benchmark sources.`;
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["adjusted_performance", "performance_rank"],
      };
    }

    // Cheapest / cost
    if (
      q.includes("cheap") ||
      q.includes("cost") ||
      q.includes("budget") ||
      q.includes("price")
    ) {
      const withCost = models.filter(
        (m) => m.blended_cost_per_1m != null && m.blended_cost_per_1m > 0,
      );
      const sorted = [...withCost]
        .sort((a, b) => a.blended_cost_per_1m - b.blended_cost_per_1m)
        .slice(0, 5);
      const answer =
        `The most affordable models in the LLMDEX index are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — $${m.blended_cost_per_1m}/1M input tokens, Performance: ${m.adjusted_performance}`,
          )
          .join("\n") +
        `\n\nNote: Lower cost doesn't always mean lower quality. Check the value leaderboard for the best cost-to-performance ratio.`;
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["blended_cost_per_1m", "adjusted_performance"],
      };
    }

    // Value / ratio
    if (q.includes("value") || q.includes("ratio") || q.includes("bang for")) {
      const withValue = models.filter((m) => m.value_rank != null);
      const sorted = [...withValue]
        .sort((a, b) => a.value_rank - b.value_rank)
        .slice(0, 5);
      const answer =
        `The best value models (performance per dollar) are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — Value Rank #${m.value_rank}, Performance: ${m.adjusted_performance}, Cost: $${m.blended_cost_per_1m || "N/A"}/1M`,
          )
          .join("\n");
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: [
          "value_rank",
          "adjusted_performance",
          "blended_cost_per_1m",
        ],
      };
    }

    // Coding
    if (
      q.includes("coding") ||
      q.includes("code") ||
      q.includes("programming") ||
      q.includes("developer")
    ) {
      const withCoding = models.filter((m) => m.coding_score != null);
      const sorted = [...withCoding]
        .sort((a, b) => b.coding_score - a.coding_score)
        .slice(0, 5);
      const answer =
        `The top models for coding tasks are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — Coding Score: ${m.coding_score}, Performance: ${m.adjusted_performance}`,
          )
          .join("\n") +
        `\n\nCoding scores are derived from benchmarks like LiveBench coding tasks and specialized programming assessments.`;
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["coding_score", "adjusted_performance"],
      };
    }

    // Reasoning
    if (
      q.includes("reason") ||
      q.includes("math") ||
      q.includes("logic") ||
      q.includes("analytical")
    ) {
      const withReason = models.filter((m) => m.reasoning_score != null);
      const sorted = [...withReason]
        .sort((a, b) => b.reasoning_score - a.reasoning_score)
        .slice(0, 5);
      const answer =
        `The top models for reasoning tasks are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — Reasoning Score: ${m.reasoning_score}, Performance: ${m.adjusted_performance}`,
          )
          .join("\n");
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["reasoning_score", "adjusted_performance"],
      };
    }

    // Context window
    if (
      q.includes("context") ||
      q.includes("window") ||
      q.includes("long document") ||
      q.includes("token")
    ) {
      const withCtx = models.filter(
        (m) => m.context_window != null && m.context_window > 0,
      );
      const sorted = [...withCtx]
        .sort((a, b) => b.context_window - a.context_window)
        .slice(0, 5);
      const answer =
        `Models with the largest context windows:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — ${(m.context_window / 1000).toFixed(0)}K tokens, Performance: ${m.adjusted_performance}`,
          )
          .join("\n");
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["context_window", "adjusted_performance"],
      };
    }

    // Compare specific models
    if (
      q.includes("compare") ||
      q.includes("vs") ||
      q.includes("versus") ||
      q.includes("difference")
    ) {
      const mentionedModels = models.filter((m) => {
        const name = m.model_name.toLowerCase();
        return (
          q.includes(name) ||
          q.split(" ").some((w) => w.length > 3 && name.includes(w))
        );
      });

      if (mentionedModels.length >= 2) {
        const answer =
          `Comparison of mentioned models:\n\n` +
          mentionedModels
            .map(
              (m) =>
                `**${m.model_name}** (${m.provider}):\n` +
                `  • Performance: ${m.adjusted_performance} (Rank #${m.performance_rank})\n` +
                `  • Cost: $${m.blended_cost_per_1m || "N/A"}/1M input\n` +
                `  • Coding: ${m.coding_score || "N/A"} | Reasoning: ${m.reasoning_score || "N/A"}\n` +
                `  • Context: ${m.context_window ? (m.context_window / 1000).toFixed(0) + "K" : "N/A"}`,
            )
            .join("\n\n");
        return {
          answer,
          referenced_models: mentionedModels.map((m) => m.model_name),
          data_points_used: [
            "adjusted_performance",
            "blended_cost_per_1m",
            "coding_score",
            "reasoning_score",
          ],
        };
      }
    }

    // Efficiency
    if (q.includes("efficien") || q.includes("speed") || q.includes("fast")) {
      const withEff = models.filter((m) => m.efficiency_rank != null);
      const sorted = [...withEff]
        .sort((a, b) => a.efficiency_rank - b.efficiency_rank)
        .slice(0, 5);
      const answer =
        `The most efficient models are:\n\n` +
        sorted
          .map(
            (m, i) =>
              `**${i + 1}. ${m.model_name}** (${m.provider}) — Efficiency Rank #${m.efficiency_rank}, Performance: ${m.adjusted_performance}`,
          )
          .join("\n");
      return {
        answer,
        referenced_models: sorted.map((m) => m.model_name),
        data_points_used: ["efficiency_rank", "adjusted_performance"],
      };
    }

    // Default: overview
    const top5 = [...models]
      .sort((a, b) => a.performance_rank - b.performance_rank)
      .slice(0, 5);
    const answer =
      `I can help you analyze model data from the LLMDEX index. Here's a quick overview of the top 5 models:\n\n` +
      top5
        .map(
          (m, i) =>
            `**${i + 1}. ${m.model_name}** (${m.provider}) — Performance: ${m.adjusted_performance}, Cost: $${m.blended_cost_per_1m || "N/A"}/1M`,
        )
        .join("\n") +
      `\n\nTry asking about:\n• Best models for coding or reasoning\n• Cheapest high-performance models\n• Context window comparisons\n• Model value rankings`;
    return {
      answer,
      referenced_models: top5.map((m) => m.model_name),
      data_points_used: ["adjusted_performance", "blended_cost_per_1m"],
    };
  }
}

//
// SORTABLE COLUMNS — Event Delegation (persistent sort state)
// ══════════════════════════════════════════════════════════════

function setupSorting(allData) {
  allDataRef = allData;

  // Use event delegation on the thead — works even after header rebuild
  const thead = document.querySelector("#models-table thead");
  if (!thead || thead.dataset.sortBound) return; // bind only once
  thead.dataset.sortBound = "true";

  thead.addEventListener("click", (e) => {
    // Ignore clicks on info icons
    if (e.target.closest(".info-icon")) return;

    const th = e.target.closest("th.sortable");
    if (!th) return;

    const field = th.dataset.sort;
    if (!field) return;

    // Toggle direction
    if (currentSort.field === field) {
      currentSort.direction = currentSort.direction === "asc" ? "desc" : "asc";
    } else {
      currentSort.field = field;
      currentSort.direction = "desc";
    }

    // Clear all sort indicators, set current
    thead
      .querySelectorAll("th")
      .forEach((h) => h.classList.remove("sort-asc", "sort-desc"));
    th.classList.add(
      currentSort.direction === "asc" ? "sort-asc" : "sort-desc",
    );

    // Sort data
    const data = allDataRef || allData;
    const sorted = [...data].sort((a, b) => {
      const va = a[field],
        vb = b[field];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return currentSort.direction === "asc" ? va - vb : vb - va;
    });

    const query = document.getElementById("search-input").value.toLowerCase();
    const provider = document.getElementById("provider-filter").value;
    const filtered = filterModels(sorted, query, provider);
    populateTable(filtered, data, currentTab);
  });
}

// ══════════════════════════════════════════════════════════════
// COLUMN TOOLTIPS
// ══════════════════════════════════════════════════════════════

function setupTooltips(columnDefs) {
  const tooltip = document.getElementById("column-tooltip");
  if (!tooltip) return;

  const closeBtn = document.getElementById("tooltip-close");
  let hideTimeout;
  let activeIcon = null;

  const hide = () => {
    tooltip.style.display = "none";
    activeIcon = null;
  };
  const scheduleHide = () => {
    hideTimeout = setTimeout(hide, 300);
  };
  const cancelHide = () => {
    if (hideTimeout) clearTimeout(hideTimeout);
  };

  closeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    hide();
  });
  window.addEventListener(
    "scroll",
    () => {
      if (tooltip.style.display === "block") hide();
    },
    { passive: true },
  );
  tooltip.addEventListener("mouseenter", cancelHide);
  tooltip.addEventListener("mouseleave", scheduleHide);

  document.addEventListener("click", (e) => {
    if (
      tooltip.style.display === "block" &&
      !tooltip.contains(e.target) &&
      !e.target.closest(".info-icon")
    ) {
      hide();
    }
  });

  const showTooltip = (icon) => {
    cancelHide();
    activeIcon = icon;
    const col = icon.dataset.col;
    const def = columnDefs[col] ||
      {
        performance: {
          label: "Adjusted Performance",
          definition:
            "Bias-corrected benchmark score based on intelligence, coding, and reasoning capability datasets.",
          formula: "performance × (0.85 + 0.15 × coverage)",
          data_sources: "LiveBench, LMSYS, Artificial Analysis",
          limitations: "Benchmarks test narrow capabilities.",
        },
        composite: {
          label: "Composite Score",
          definition:
            "Weighted value score combining pure capabilities with cost and throughput considerations.",
          formula: "50% Perf + 30% Cost Eff + 20% Speed",
          data_sources: "LLMDEX Aggregated Pipeline",
          limitations:
            "Weights are generalized and may not fit specific use-cases.",
        },
        cost_input: {
          label: "Input Cost",
          definition: "Stated API price per 1,000,000 tokens of input/prompt.",
          formula: "Real-world API cost",
          data_sources: "Provider Documentation",
          limitations:
            "Does not account for volume discounts or exact tokenization differences.",
        },
        cost_output: {
          label: "Output Cost",
          definition:
            "Stated API price per 1,000,000 tokens of output/completion.",
          formula: "Real-world API cost",
          data_sources: "Provider Documentation",
          limitations: "Does not account for volume discounts.",
        },
        context: {
          label: "Context Window",
          definition: "Maximum input token capacity supported by the model.",
          formula: "Context token limit",
          data_sources: "Provider Documentation",
          limitations:
            "Some models degrade in recall accuracy at extreme context sizes.",
        },
      }[col] || {
        label: col.charAt(0).toUpperCase() + col.slice(1).replace(/_/g, " "),
        definition: "Information unavailable",
        formula: "N/A",
        data_sources: "N/A",
        limitations: "N/A",
      };

    document.getElementById("tooltip-title").textContent = def.label || col;
    document.getElementById("tooltip-definition").textContent =
      def.definition || "—";
    document.getElementById("tooltip-formula").textContent = def.formula || "—";
    document.getElementById("tooltip-sources").textContent =
      def.data_sources || "—";
    document.getElementById("tooltip-limitations").textContent =
      def.limitations || "—";

    tooltip.style.display = "block";
    tooltip.classList.remove("tooltip-above");

    const iconRect = icon.getBoundingClientRect();
    const tooltipW = tooltip.offsetWidth;
    const tooltipH = tooltip.offsetHeight;
    const gap = 12;
    const vpW = window.innerWidth;
    const vpH = window.innerHeight;

    const iconCenterX = iconRect.left + iconRect.width / 2;
    let left = iconCenterX - tooltipW / 2;
    left = Math.max(12, Math.min(left, vpW - tooltipW - 12));

    const arrowLeft = Math.max(20, Math.min(iconCenterX - left, tooltipW - 20));
    tooltip.style.setProperty("--arrow-left", `${arrowLeft}px`);

    const spaceBelow = vpH - iconRect.bottom - gap;
    const spaceAbove = iconRect.top - gap;
    let top;
    if (spaceBelow >= tooltipH || spaceBelow >= spaceAbove) {
      top = iconRect.bottom + gap;
      tooltip.classList.remove("tooltip-above");
    } else {
      top = iconRect.top - tooltipH - gap;
      tooltip.classList.add("tooltip-above");
    }
    top = Math.max(8, Math.min(top, vpH - tooltipH - 8));

    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
  };

  // Use event delegation for dynamic info icons
  document.addEventListener(
    "mouseenter",
    (e) => {
      const icon = e.target.closest(".info-icon");
      if (icon) showTooltip(icon);
    },
    true,
  );

  document.addEventListener(
    "mouseleave",
    (e) => {
      const icon = e.target.closest(".info-icon");
      if (icon) scheduleHide();
    },
    true,
  );

  document.addEventListener("click", (e) => {
    const icon = e.target.closest(".info-icon");
    if (icon) {
      e.stopPropagation();
      if (activeIcon === icon && tooltip.style.display === "block") {
        hide();
      } else {
        showTooltip(icon);
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// MODEL COMPARISON
// ══════════════════════════════════════════════════════════════

function getSelectedSlugs() {
  return Array.from(document.querySelectorAll(".model-checkbox:checked"))
    .map((cb) => cb.dataset.slug)
    .filter((s) => s && s.length > 0);
}

function getModelDisplayName(slug) {
  if (!allDataRef) return slug;
  const m = allDataRef.find(
    (d) =>
      d.model_slug === slug ||
      d.canonical_name === slug ||
      d.model_name === slug,
  );
  return m ? m.canonical_name || m.model_name || slug : slug;
}

function updateCompareBar() {
  const slugs = getSelectedSlugs();
  const bar = document.getElementById("compare-floating-bar");
  const countEl = document.getElementById("compare-count");
  const inlineBtn = document.getElementById("compare-btn");
  const selectedList = document.getElementById("compare-selected-models");
  const goBtn = document.getElementById("compare-go-btn");

  // Update the inline button count
  if (countEl) countEl.textContent = slugs.length;
  if (inlineBtn) inlineBtn.disabled = slugs.length < 2;

  // Show/hide floating bar
  if (bar) {
    if (slugs.length > 0) {
      bar.classList.add("visible");
    } else {
      bar.classList.remove("visible");
    }
  }

  // Render selected model chips
  if (selectedList) {
    selectedList.innerHTML = slugs
      .map((slug) => {
        const name = getModelDisplayName(slug);
        return `<span class="compare-chip">
        ${name}
        <button class="compare-chip-remove" data-slug="${slug}" title="Remove">×</button>
      </span>`;
      })
      .join("");

    // Bind remove buttons
    selectedList.querySelectorAll(".compare-chip-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const slug = btn.dataset.slug;
        const cb = document.querySelector(
          `.model-checkbox[data-slug="${slug}"]`,
        );
        if (cb) {
          cb.checked = false;
        }
        updateCompareBar();
      });
    });
  }

  // Enable/disable go button
  if (goBtn) {
    goBtn.disabled = slugs.length < 2;
    goBtn.querySelector(".compare-go-count").textContent = slugs.length;
  }
}

function setupComparison(allData) {
  const section = document.getElementById("comparison-section");
  const closeBtn = document.getElementById("close-comparison");
  if (!section) return;

  // Inject floating compare bar into page
  const bar = document.createElement("div");
  bar.id = "compare-floating-bar";
  bar.className = "compare-floating-bar glass";
  bar.innerHTML = `
    <div class="compare-bar-content">
      <div class="compare-bar-left">
        <i class="fas fa-columns compare-bar-icon"></i>
        <span class="compare-bar-label">Compare Models</span>
        <div id="compare-selected-models" class="compare-selected-models"></div>
      </div>
      <div class="compare-bar-right">
        <button class="btn secondary compare-clear-btn" id="compare-clear-btn">
          <i class="fas fa-times"></i> Clear
        </button>
        <button class="btn primary compare-go" id="compare-go-btn" disabled>
          <i class="fas fa-chart-radar"></i> Compare <span class="compare-go-count">0</span>
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(bar);

  // Delegation on tbody for checkbox changes
  document
    .querySelector("#models-table tbody")
    .addEventListener("change", (e) => {
      if (e.target.classList.contains("model-checkbox")) updateCompareBar();
    });

  // Also handle the inline compare button
  const inlineBtn = document.getElementById("compare-btn");
  if (inlineBtn) {
    inlineBtn.addEventListener("click", () =>
      triggerComparison(allData, section),
    );
  }

  // Floating bar go button
  document.getElementById("compare-go-btn").addEventListener("click", () => {
    triggerComparison(allData, section);
  });

  // Clear button
  document.getElementById("compare-clear-btn").addEventListener("click", () => {
    document.querySelectorAll(".model-checkbox:checked").forEach((cb) => {
      cb.checked = false;
    });
    const selectAll = document.getElementById("select-all-models");
    if (selectAll) selectAll.checked = false;
    updateCompareBar();
  });

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      section.style.display = "none";
    });
  }
}

function triggerComparison(allData, section) {
  const slugs = getSelectedSlugs();
  if (slugs.length < 2) return;

  const findModel = (slug) => {
    return allData.find(
      (d) =>
        d.model_slug === slug ||
        d.canonical_name === slug ||
        d.model_name === slug,
    );
  };

  const models = slugs.map(findModel).filter(Boolean);
  if (models.length < 2) return;

  section.style.display = "block";
  renderComparisonRadar(models);
  section.scrollIntoView({ behavior: "smooth" });
}

function renderComparisonRadar(models) {
  const colors = ["#3b82f6", "#f59e0b", "#10b981"];
  const fills = [
    "rgba(59,130,246,0.1)",
    "rgba(245,158,11,0.1)",
    "rgba(16,185,129,0.1)",
  ];

  const axesDef = [
    { name: "Performance", key: "adjusted_performance" },
    { name: "Cost Efficiency", key: "cost_index" },
    { name: "Coding", key: "coding_score" },
    { name: "Reasoning", key: "reasoning_score" },
    { name: "Speed", key: "speed_index" },
  ];

  const traces = models.map((model, i) => {
    const values = axesDef.map((a) => {
      const v = model[a.key];
      return v != null ? v : 0;
    });
    values.push(values[0]);
    const labels = axesDef.map((a) => a.name);
    labels.push(labels[0]);

    return {
      type: "scatterpolar",
      r: values,
      theta: labels,
      fill: "toself",
      fillcolor: fills[i],
      line: { color: colors[i] },
      name: model.canonical_name || model.model_name,
    };
  });

  const isMobile = window.innerWidth <= 768;
  const layout = {
    title: {
      text: "Model Comparison",
      font: { color: "#f8fafc", size: isMobile ? 13 : 18 },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    margin: isMobile ? { t: 45, b: 65, l: 40, r: 40 } : { t: 60, b: 30 },
    polar: {
      radialaxis: {
        visible: true,
        range: [0, 100],
        color: "#94a3b8",
        tickfont: { size: isMobile ? 6 : 12 },
      },
      bgcolor: "rgba(0,0,0,0)",
      angularaxis: {
        color: "#f8fafc",
        tickfont: { size: isMobile ? 7 : 12 },
      },
    },
    showlegend: true,
    legend: {
      font: { color: "#f8fafc", size: isMobile ? 9 : 12 },
      orientation: isMobile ? "h" : "v",
      y: isMobile ? -0.3 : undefined,
      x: isMobile ? 0.5 : undefined,
      xanchor: isMobile ? "center" : undefined,
    },
  };

  Plotly.newPlot("comparison-radar", traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ══════════════════════════════════════════════════════════════
// FAMILY GROWTH EXPLORER (Part 8 implementation)
// ══════════════════════════════════════════════════════════════

/**
 * Detect sub-family from model name (mirrors Python pipeline logic).
 */
function detectSubFamily(name, provider) {
  const n = name.toLowerCase();
  if (n.includes("claude")) {
    if (n.includes("opus")) return "Claude Opus";
    if (n.includes("sonnet")) return "Claude Sonnet";
    if (n.includes("haiku")) return "Claude Haiku";
    return "Claude Other";
  }
  if (n.includes("gemini")) {
    if (n.includes("pro")) return "Gemini Pro";
    if (n.includes("flash")) return "Gemini Flash";
    if (n.includes("ultra")) return "Gemini Ultra";
    if (n.includes("nano")) return "Gemini Nano";
    return "Gemini Other";
  }
  if (n.includes("gpt")) {
    if (n.includes("mini")) return "GPT Mini";
    if (n.includes("medium")) return "GPT Medium";
    if (n.includes("high")) return "GPT High";
    if (n.includes("codex")) return "GPT Codex";
    if (n.includes("pro")) return "GPT Pro";
    if (n.includes("oss")) return "GPT OSS";
    return "GPT Other";
  }
  if (n.includes("deepseek")) {
    if (n.includes("r1")) return "DeepSeek R";
    if (n.includes("speciale")) return "DeepSeek Speciale";
    return "DeepSeek V";
  }
  if (n.includes("qwen")) {
    if (n.includes("max")) return "Qwen Max";
    if (n.includes("plus")) return "Qwen Plus";
    return "Qwen Base";
  }
  if (n.includes("kimi")) return "Kimi";
  if (n.includes("glm")) return "GLM";
  if (n.includes("grok")) return "Grok";
  if (n.includes("llama")) return "Meta Llama";
  if (n.includes("minimax")) return "MiniMax";
  if (n.includes("mimo")) return "MiMo";
  if (
    provider &&
    !["unknown", "other", "null"].includes(provider.toLowerCase())
  )
    return provider;
  return name.split(" ")[0] || "Unknown";
}

/**
 * Extract the brand (top-level company) from a sub-family name.
 * e.g. "Claude Opus" → "Claude", "Gemini Flash" → "Gemini", "GPT High" → "GPT"
 */
function getBrandFromSubFamily(subFamily) {
  const brandMap = {
    Claude: [
      "Claude Opus",
      "Claude Sonnet",
      "Claude Haiku",
      "Claude Other",
      "Claude Base",
    ],
    Gemini: [
      "Gemini Pro",
      "Gemini Flash",
      "Gemini Ultra",
      "Gemini Nano",
      "Gemini Other",
      "Gemini Base",
    ],
    GPT: [
      "GPT Mini",
      "GPT Medium",
      "GPT High",
      "GPT Codex",
      "GPT Pro",
      "GPT OSS",
      "GPT Other",
      "GPT Base",
    ],
    DeepSeek: [
      "DeepSeek R",
      "DeepSeek Speciale",
      "DeepSeek V",
      "DeepSeek Base",
    ],
    Qwen: ["Qwen Max", "Qwen Plus", "Qwen Base"],
  };
  for (const [brand, subs] of Object.entries(brandMap)) {
    if (subs.includes(subFamily)) return brand;
  }
  // For single-name families like "Kimi", "GLM", the sub-family IS the brand
  return subFamily;
}

/**
 * Build family history from live index data (client-side fallback).
 */
function buildFamiliesFromIndex(allData) {
  const families = {};
  allData.forEach((row) => {
    const name = row.canonical_name || row.model_name;
    if (!name) return;
    const perf = row.adjusted_performance ?? row.performance_index;
    if (perf == null) return;
    const fam = detectSubFamily(name, row.provider);
    if (!families[fam]) families[fam] = [];
    families[fam].push({
      name,
      performance: perf,
      rank: row.performance_rank,
      provider: row.provider,
    });
  });
  const result = {};
  for (const [fam, members] of Object.entries(families)) {
    if (members.length < 2) continue;
    members.sort((a, b) => a.performance - b.performance);
    result[fam] = members;
  }
  return result;
}

// Vibrant, modern glassmorphism palette — highly distinguishable and premium
const SUB_FAMILY_COLORS = [
  { fill: "rgba(45, 212, 191, 0.95)", line: "#14b8a6" }, // teal
  { fill: "rgba(167, 139, 250, 0.95)", line: "#8b5cf6" }, // purple
  { fill: "rgba(251, 146, 60, 0.95)", line: "#f97316" }, // orange
  { fill: "rgba(56, 189, 248, 0.95)", line: "#0ea5e9" }, // sky
  { fill: "rgba(244, 114, 182, 0.95)", line: "#ec4899" }, // pink
  { fill: "rgba(250, 204, 21, 0.95)", line: "#eab308" }, // yellow
  { fill: "rgba(163, 230, 53, 0.95)", line: "#84cc16" }, // lime
];

async function setupFamilyExplorer(allData) {
  const chartDiv = document.getElementById("family-chart");
  const selector = document.getElementById("family-selector");
  const explanation = document.querySelector(".family-explanation-text");

  if (!chartDiv || !selector) return;

  // 1. Load sub-family data
  let subFamilyData = {};
  let dataSource = "static";
  try {
    const res = await fetch("../data/history/family_growth.json");
    if (res.ok) {
      subFamilyData = await res.json();
    }
  } catch (e) {
    console.warn("family_growth.json not available:", e);
  }

  // Fallback: build from live index data
  if (
    Object.keys(subFamilyData).length === 0 &&
    allData &&
    allData.length > 0
  ) {
    console.log("Building family history from live index data...");
    subFamilyData = buildFamiliesFromIndex(allData);
    dataSource = "live";
  }

  if (Object.keys(subFamilyData).length === 0) {
    chartDiv.innerHTML =
      "<p class='error-msg'>No family history data available.</p>";
    return;
  }

  // 2. Group sub-families into brands
  //    brandGroups = { "Claude": { "Claude Opus": [...], "Claude Sonnet": [...] }, ... }
  const brandGroups = {};
  for (const [subFam, members] of Object.entries(subFamilyData)) {
    const brand = getBrandFromSubFamily(subFam);
    if (!brandGroups[brand]) brandGroups[brand] = {};
    brandGroups[brand][subFam] = members;
  }

  // Only show brands that have at least 2 sub-families (so there's something to compare)
  // OR brands with 1 sub-family that has multiple variants
  const brands = Object.keys(brandGroups).sort();

  if (brands.length === 0) {
    chartDiv.innerHTML = "<p class='error-msg'>No family data available.</p>";
    return;
  }

  // 3. Populate Dropdown with brand names
  selector.innerHTML = '<option value="">Select a company / brand...</option>';

  brands.forEach((brand) => {
    const subFams = Object.keys(brandGroups[brand]);
    const totalVariants = Object.values(brandGroups[brand]).reduce(
      (sum, m) => sum + m.length,
      0,
    );
    const opt = document.createElement("option");
    opt.value = brand;
    opt.textContent = `${brand} (${subFams.length} ${subFams.length > 1 ? "families" : "family"}, ${totalVariants} models)`;
    selector.appendChild(opt);
  });

  // 4. Render Chart Function
  const renderChart = (brand) => {
    // If no brand selected, show empty state
    if (!brand || !brandGroups[brand]) {
      chartDiv.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;min-height:400px;color:var(--text-secondary);text-align:center;">
           <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 50%; margin-bottom: 1.5rem; box-shadow: 0 4px 30px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.05);">
             <i class="fas fa-chart-bar" style="font-size:3.5rem;color:var(--accent-glow);opacity:0.8;"></i>
           </div>
           <h3 style="color:#f8fafc; margin-bottom: 0.5rem; font-size: 1.3rem;">Compare Model Families</h3>
           <p style="font-size:1rem; max-width: 340px; opacity: 0.8; line-height: 1.5;">Select a company from the dropdown to compare its model families side by side — Opus vs Sonnet vs Haiku, Pro vs Flash, and more.</p>
        </div>`;
      if (explanation) explanation.style.display = "block";
      return;
    }

    const subFams = brandGroups[brand];
    const subFamNames = Object.keys(subFams).sort();

    // For each sub-family, pick the BEST performing variant as the representative,
    // plus show all variants as bars grouped by sub-family
    const traces = [];

    // Build: each sub-family becomes a trace (group) in the grouped bar chart
    // X-axis: sub-family names (Opus, Sonnet, Haiku)
    // We'll show ALL variants per sub-family as individual bars

    // Collect all variants across all sub-families, tagged with their sub-family
    const allVariants = [];
    subFamNames.forEach((sf, sfIdx) => {
      const members = [...subFams[sf]].sort(
        (a, b) => a.performance - b.performance,
      );
      members.forEach((m) => {
        allVariants.push({
          ...m,
          subFamily: sf,
          subFamilyIdx: sfIdx,
          // Extract just the sub-family type (e.g. "Opus" from "Claude Opus")
          subFamilyShort: sf.replace(brand + " ", "") || sf,
        });
      });
    });

    // Sort all variants by sub-family then by performance
    allVariants.sort((a, b) => {
      if (a.subFamilyIdx !== b.subFamilyIdx)
        return a.subFamilyIdx - b.subFamilyIdx;
      return a.performance - b.performance;
    });

    // Clean label: transform full dates into (MM/DD) to avoid identical overlapping bars
    function cleanLabel(name) {
      let s = name
        .replace(/\s*\(?20\d{2}-(\d{2})-(\d{2})\)?/g, " ($1/$2)")
        .replace(/\(Max Thinking\)/gi, "(Think.)")
        .replace(/\(Thinking\)/gi, "(Think.)")
        .replace(/\bNEW\b/gi, "")
        .replace(/\bPreview\b/gi, "")
        .trim();
      // Strip brand prefix for brevity
      const brandParts = brand.split(" ");
      brandParts.forEach((p) => {
        const regex = new RegExp("\\b" + p + "\\b", "gi");
        s = s.replace(regex, "");
      });
      s = s.replace(/\s+/g, " ").trim();
      if (s.length === 0) s = name;
      return s.length > 25 ? s.slice(0, 24) + "…" : s;
    }

    // We will use a single trace for all bars to keep them perfectly centered and wide,
    // and dummy traces for the legend items.
    const xLabelsAll = [];
    const yValuesAll = [];
    const hoverTextsAll = [];
    const markerColorsAll = [];
    const markerLineColorsAll = [];

    // Create one dummy trace per sub-family for legend + color grouping,
    // and collect all bars into the main arrays.
    subFamNames.forEach((sf, sfIdx) => {
      const members = [...subFams[sf]].sort(
        (a, b) => a.performance - b.performance,
      );
      const color = SUB_FAMILY_COLORS[sfIdx % SUB_FAMILY_COLORS.length];
      const shortName = sf.replace(brand + " ", "") || sf;

      members.forEach((m) => {
        xLabelsAll.push(cleanLabel(m.name));
        yValuesAll.push(m.performance);
        markerColorsAll.push(color.fill);
        markerLineColorsAll.push(color.line);

        const rank = m.rank != null ? `#${m.rank}` : "N/A";
        let impStr = "";
        if (m.predecessor) {
          const sign = m.improvement_abs >= 0 ? "+" : "";
          impStr = `<br>Change: <b>${sign}${m.improvement_abs.toFixed(1)} pts</b> (${sign}${m.improvement_pct}%)`;
        }
        hoverTextsAll.push(
          `<b>${m.name}</b><br>` +
            `Family: <b>${shortName}</b><br>` +
            `Performance: <b>${m.performance.toFixed(1)}</b><br>` +
            `Global Rank: ${rank}${impStr}`,
        );
      });

      // Add dummy trace for the legend
      traces.push({
        x: [null],
        y: [null],
        type: "scatter",
        mode: "markers",
        name: shortName,
        marker: {
          color: color.fill,
          symbol: "square",
          size: 15,
          line: { color: color.line, width: 1 },
        },
        hoverinfo: "none",
        showlegend: true,
      });
    });

    // Add the main unified trace containing all bars perfectly centered
    traces.push({
      x: xLabelsAll,
      y: yValuesAll,
      type: "bar",
      name: "Models",
      marker: {
        color: markerColorsAll,
        line: { color: markerLineColorsAll, width: 1 },
        cornerradius: 4,
      },
      text: yValuesAll.map((v) => v.toFixed(1)),
      textposition: "outside",
      constraintext: "none",
      textfont: {
        color: "#ffffff",
        size: 13,
        family: "Inter, system-ui, sans-serif",
      },
      hovertext: hoverTextsAll,
      hoverinfo: "text",
      cliponaxis: false,
      showlegend: false,
    });

    // Calculate Y range — generous headroom for outside text labels
    const allPerfs = allVariants.map((v) => v.performance);
    const minY = Math.min(...allPerfs);
    const maxY = Math.max(...allPerfs);
    const spread = maxY - minY;
    const rangeFloor = Math.max(0, minY - (spread > 0 ? spread * 0.25 : 10));
    const rangeCeil = maxY + (spread > 0 ? spread * 0.35 : 8);

    const isMobile = window.innerWidth <= 768;
    const layout = {
      title: {
        text: `<b>${brand}</b>${isMobile ? "" : " — Family Comparison"}`,
        font: {
          color: "#f8fafc",
          size: isMobile ? 15 : 18,
          family: "Inter, system-ui, sans-serif",
        },
        y: 0.97,
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", color: "#94a3b8" },
      xaxis: {
        gridcolor: "rgba(255,255,255,0.04)",
        tickangle: isMobile ? -45 : 0,
        automargin: true,
        showgrid: false,
        tickfont: { size: isMobile ? 10 : 13, color: "#e2e8f0" },
        type: "category",
        categoryorder: "array",
        categoryarray: xLabelsAll,
        tickpad: isMobile ? 8 : 14,
        linecolor: "rgba(255,255,255,0.06)",
      },
      yaxis: {
        title: {
          text: isMobile ? "Score" : "Performance Score",
          font: { color: "#94a3b8", size: isMobile ? 10 : 11 },
        },
        gridcolor: "rgba(255,255,255,0.05)",
        zerolinecolor: "rgba(255,255,255,0.06)",
        range: [rangeFloor, rangeCeil],
        tickfont: { size: isMobile ? 10 : 11, color: "#94a3b8" },
      },
      margin: isMobile
        ? { t: 40, r: 15, b: 120, l: 45 }
        : { t: 50, r: 20, b: 140, l: 55 },
      showlegend: true,
      legend: {
        font: {
          color: "#e2e8f0",
          size: 13,
          family: "Inter, system-ui, sans-serif",
        },
        bgcolor: "rgba(0,0,0,0)",
        orientation: "h",
        y: -0.3,
        x: 0.5,
        xanchor: "center",
      },
      hovermode: "closest",
      bargap: 0.15,
      height: 550,
    };

    chartDiv.innerHTML = "";

    // Make it horizontally scrollable: dynamically scale width based on data
    const numBars = xLabelsAll.length;
    const padding = 100;
    const pixelsPerBar = 150;
    const minWidthNeeded = numBars * pixelsPerBar + padding;
    const parentWidth = chartDiv.parentElement
      ? chartDiv.parentElement.offsetWidth
      : 800;

    layout.width = Math.max(parentWidth, minWidthNeeded);

    chartDiv.style.overflowX = "auto";
    chartDiv.style.overflowY = "hidden";

    Plotly.newPlot(chartDiv, traces, layout, {
      displayModeBar: true,
      scrollZoom: true,
      responsive: false,
    });

    // Caption
    const existingCap = chartDiv.querySelector(".family-chart-caption");
    if (!existingCap) {
      const cap = document.createElement("p");
      cap.className = "family-chart-caption";
      cap.style.cssText =
        "text-align:center;color:#94a3b8;font-size:0.8rem;margin-top:0.5rem;opacity:0.7;";
      const totalModels = allVariants.length;
      cap.textContent = `Comparing ${subFamNames.length} families, ${totalModels} models. Color-coded by family. Hover for details.`;
      chartDiv.appendChild(cap);
    }
  };

  // 5. Event Listener
  selector.addEventListener("change", (e) => {
    renderChart(e.target.value);
  });

  // Initial empty state
  renderChart("");
}
