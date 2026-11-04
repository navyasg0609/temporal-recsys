// One load step: VUS users each ask for a recommendation list, then wait about a second, like
// a person clicking. The first 10 seconds warm up and are tagged out of the measurement.
import http from "k6/http";
import { check, sleep } from "k6";
import { SharedArray } from "k6/data";
import exec from "k6/execution";

const WARMUP_MS = 10000;
const users = new SharedArray("users", () => JSON.parse(open("/load/users.json")));

export const options = {
  discardResponseBodies: true,
  summaryTrendStats: ["med", "p(95)", "p(99)", "max"],
  scenarios: {
    step: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS),
      duration: `${WARMUP_MS / 1000 + Number(__ENV.SECONDS)}s`,
      gracefulStop: "5s",
    },
  },
  // Empty thresholds make k6 report these measured-only submetrics in the summary.
  thresholds: {
    "http_req_duration{phase:measured}": [],
    "http_req_failed{phase:measured}": [],
    "http_reqs{phase:measured}": [],
  },
};

export default function () {
  const phase = exec.instance.currentTestRunDuration < WARMUP_MS ? "warmup" : "measured";
  const user = users[Math.floor(Math.random() * users.length)];
  const response = http.get(`${__ENV.TARGET}/recommendations/${user}?k=10`, {
    timeout: "5s",
    tags: { phase },
  });
  check(response, { "status is 200": (r) => r.status === 200 });
  // A random wait keeps users out of step. A fixed one makes them all fire at once.
  sleep(0.5 + Math.random());
}

export function handleSummary(data) {
  return { [__ENV.OUT]: JSON.stringify(data.metrics) };
}
