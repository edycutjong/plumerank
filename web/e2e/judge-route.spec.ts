import { test, expect } from "@playwright/test";

/**
 * /judge exists for exactly one reader. This test is the mechanical half of that
 * promise (JUDGE.md section 1 / enhance-project.md section 3.9): fetched with no
 * credentials and no session, it must return 200 and carry the claim sentence.
 */

test("GET /judge returns 200 with no credentials or session and carries the claim", async ({
  browser,
}) => {
  const context = await browser.newContext(); // fresh context: no cookies, no storage
  const page = await context.newPage();

  const response = await page.goto("/judge");
  expect(response?.status()).toBe(200);

  await expect(
    page.getByText(/calibrated to a fixed budget of 2 alerts per camera[-\u2011]window/i)
  ).toBeVisible();

  // The 30-second path, the receipt, the reproduce command, and the honest-limitations
  // section must all be reachable on this one page without navigating away.
  await expect(page.getByText(/30[-\u2011]second path/i)).toBeVisible();
  await expect(page.getByText(/the receipt/i)).toBeVisible();
  await expect(page.getByText(/make verify/)).toBeVisible();
  await expect(page.getByText(/honest limitations/i)).toBeVisible();

  await context.close();
});
