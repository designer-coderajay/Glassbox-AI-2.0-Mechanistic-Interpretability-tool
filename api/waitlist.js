export const config = {
  runtime: 'edge',
};

const ALLOWED_ORIGINS = [
  'https://project-gu05p.vercel.app',
  'https://glassbox-ai.vercel.app',
];

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

  const { email, name, company, plan } = body || {};

  // Validate required field
  if (!email || !isValidEmail(email)) {
    return jsonResponse(
      { success: false, error: 'A valid email address is required.' },
      400,
      cors,
    );
  }

  // Build the submission record (KV-compatible structure).
  // When Vercel KV is configured via the dashboard, swap the comment below
  // with: await kv.hset(`waitlist:${email}`, submission);
  const submission = {
    email: email.trim().toLowerCase(),
    name: name ? String(name).trim() : null,
    company: company ? String(company).trim() : null,
    plan: plan ? String(plan).trim() : null,
    submittedAt: new Date().toISOString(),
  };

  // --- KV storage placeholder ---
  // To persist data, add Vercel KV in the dashboard and uncomment:
  //
  //   import { kv } from '@vercel/kv';
  //   await kv.hset(`waitlist:${submission.email}`, submission);
  //   await kv.sadd('waitlist:emails', submission.email);
  //
  // For now we log the submission (visible in Vercel Function logs).
  console.log('[waitlist] new submission:', JSON.stringify(submission));

  return jsonResponse(
    {
      success: true,
      message: "You're on the list! We'll reach out before August 2026.",
    },
    200,
    cors,
  );
}
