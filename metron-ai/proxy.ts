import { NextRequest, NextResponse } from 'next/server'

export default function proxy(req: NextRequest) {
  const session = req.cookies.get('metron_session')

  if (req.nextUrl.pathname.startsWith('/dashboard') && !session) {
    return NextResponse.redirect(new URL('/', req.nextUrl))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/dashboard', '/dashboard/:path*'],
}
