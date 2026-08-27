# What Is a VPN

A virtual private network (VPN) is a technology that creates an encrypted
connection, often called a tunnel, between a device and a private
network over the public internet. Traffic sent through the tunnel is
encrypted, so an observer on the intervening network path sees only
opaque encrypted packets rather than the underlying data.

## Why Organizations Use VPNs

Organizations commonly use VPNs to let remote employees reach internal
systems, such as file servers or internal applications, as though they
were physically on the corporate network. A VPN client authenticates the
user (often combined with multi-factor authentication) before granting
access to the tunnel.

## Common VPN Protocols

Widely used VPN protocols include IPsec, OpenVPN, and WireGuard. They
differ in performance characteristics and cryptographic design, but all
serve the same basic purpose: authenticating a connection and encrypting
the traffic that flows through it.

## Split Tunneling

Some VPN configurations use "split tunneling," where only traffic bound
for the private network goes through the encrypted tunnel and other
internet traffic goes directly out the user's normal connection. This
improves performance for general browsing but must be configured
carefully so sensitive traffic is not accidentally routed outside the
tunnel.
