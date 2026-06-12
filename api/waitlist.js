export const config = {
  runtime: 'edge',
};

const ALLOWED_ORIGINS = [
  'https://repo-ashen-psi.vercel.app',
  'https://project-gu05p.vercel.app',
  'https://glassbox-ai.vercel.app',
];

// Where signup notifications are delivered. Override with NOTIFY_EMAIL env var.
const DEFAULT_NOTIFY_EMAIL = 'mahale.ajay01@gmail.com';

function corsHeaders(origin) {
  const allowedOrigin = ALLOWED_ORIGINS.includes(origin)
    ? origin
    : ALLOWED_ORIGINS[0];

  return {
    'Access-Control-Allow-Origin': allowedOrigin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
  };
}

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...extraHeaders,
    },
  });
}

function isValidEmail(email) {
  return typeof email === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Email the founder about a new signup via Resend (https://resend.com).
 * Requires the RESEND_API_KEY env var in the Vercel project settings.
 * Free tier (100 emails/day) is more than enough for waitlist volume.
 * Returns true on success, false otherwise — never throws.
 */
async function notifyByEmail(submission) {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) return false;

  const to = process.env.NOTIFY_EMAIL || DEFAULT_NOTIFY_EMAIL;
  // 'onboarding@resend.dev' works without domain verification.
  // After verifying a domain in Resend, set RESEND_FROM to e.g.
  // 'Glassbox Waitlist <waitlist@yourdomain.com>'.
  const from = process.env.RESEND_FROM || 'Glassbox Waitlist <onboarding@resend.dev>';

  const rows = Object.entries(submission)
    .map(([k, v]) => `<tr><td style="padding:4px 12px 4px 0;color:#666">${escapeHtml(k)}</td><td style="padding:4px 0"><b>${escapeHtml(v ?? '—')}</b></td></tr>`)
    .join('');

  try {
    const res = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from,
        to: [to],
        reply_to: submission.email,
        subject: `Glassbox waitlist: ${submission.email}`,
        html: `<h2 style="font-family:sans-serif">New waitlist signup</h2>
<table style="font-family:sans-serif;font-size:14px">${rows}</table>
<p style="font-family:sans-serif;font-size:12px;color:#999">Reply to this email to contact them directly (reply-to is set to the registrant).</p>`,
      }),
    });
    if (!res.ok) {
      console.error('[waitlist] Resend API error:', res.status, await res.text());
      return false;
    }
    return true;
  } catch (err) {
    console.error('[waitlist] email notification failed:', err);
    return false;
  }
}

/**
 * Persist the signup to Vercel KV (Upstash Redis) via its REST API.
 * Active automatically once a KV store is attached to the project
 * (Vercel injects KV_REST_API_URL / KV_REST_API_TOKEN).
 * Returns true on success, false otherwise — never throws.
 */
async function persistToKV(submission) {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return false;

  try {
    const res = await fetch(`${url}/pipeline`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([
        ['SET', `waitlist:${submission.email}`, JSON.stringify(submission)],
        ['SADD', 'waitlist:emails', submission.email],
      ]),
    });
    if (!res.ok) {
      console.error('[waitlist] KV error:', res.status, await res.text());
      return false;
    }
    return true;
  } catch (err) {
    console.error('[waitlist] KV persistence failed:', err);
    return false;
  }
}

export default async function handler(request) {
  const origin = request.headers.get('origin') || '';
  const cors = corsHeaders(origin);

  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: cors });
  }

  if (request.method !== 'POST') {
    return jsonResponse(
      { success: false, error: 'Method not allowed. Use POST.' },
      405,
      cors,
    );
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse(
      { success: false, error: 'Invalid JSON body.' },
      400,
      cors,
    );
  }

  const { email, name, company, plan, message } = body || {};

  // Validate required field
  if (!email || !isValidEmail(email)) {
    return jsonResponse(
      { success: false, error: 'A valid email address is required.' },
      400,
      cors,
    );
  }

  const submission = {
    email: email.trim().toLowerCase(),
    name: name ? String(name).trim().slice(0, 200) : null,
    company: company ? String(company).trim().slice(0, 200) : null,
    plan: plan ? String(plan).trim().slice(0, 50) : null,
    message: message ? String(message).trim().slice(0, 2000) : null,
    submittedAt: new Date().toISOString(),
  };

  // Always log (visible in Vercel → project → Logs, filter "[waitlist]").
  console.log('[waitlist] new submission:', JSON.stringify(submission));

  // Best-effort delivery + storage. Neither failure blocks the signup.
  const [notified, stored] = await Promise.all([
    notifyByEmail(submission),
    persistToKV(submission),
  ]);
  console.log(`[waitlist] notified=${notified} stored=${stored}`);

  return jsonResponse(
    {
      success: true,
      message: "You're on the list! We'll reach out before Pro opens.",
    },
    200,
    cors,
  );
}
