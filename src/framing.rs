//! Application-layer framing above the physical packet.
//!
//! Every frame names its sender **and** its intended recipient, so a receiver
//! can tell "this is for me, and I know who it is from" with no out-of-band
//! state. That is what makes stop-and-wait ARQ workable on a shared acoustic
//! channel: an ACK is trusted only when it carries the peer id we are talking to.
//!
//! Byte layouts are big-endian and must match the Python reference exactly:
//!
//! ```text
//! data : type(1) src(2) dst(2) msg_id(2) frag_idx(2) frag_total(2)  = 11 B
//! hello: type(1) src(2) dst(2)                                      =  5 B
//! ack  : type(1) src(2) dst(2) msg_id(2) frag_idx(2)                =  9 B
//! ```

pub const FRAG_HDR_LEN: usize = 11;
pub const HELLO_HDR_LEN: usize = 5;
pub const ACK_HDR_LEN: usize = 9;

pub const T_TEXT: u8 = 0x01;
pub const T_FILE: u8 = 0x02;
/// Explicit discovery. Not used before a send — the first data fragment already
/// introduces us — but kept for "is anyone out there?" without a transfer.
pub const T_HELLO: u8 = 0x10;
pub const T_HELLO_ACK: u8 = 0x11;
pub const T_ACK: u8 = 0x12;

/// High bit of the type byte: "acknowledge this frame".
///
/// A data frame that asks to be acked *is* the introduction: it already carries
/// our id, and the ack carries the peer's. A separate hello round trip before
/// every message would double the air time of a short one to learn nothing new.
pub const ACK_REQ: u8 = 0x80;
pub const TYPE_MASK: u8 = 0x7F;

/// The dst everyone accepts and, absent [`ACK_REQ`], nobody acknowledges.
pub const BROADCAST: u16 = 0x0000;

/// Bytes per fragment. Small keeps frames short and cheap to retry.
pub const FRAG_PAYLOAD: usize = 48;

pub type DeviceId = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragHeader {
    pub kind: u8,
    pub ack_req: bool,
    pub src: DeviceId,
    pub dst: DeviceId,
    pub msg_id: u16,
    pub frag_idx: u16,
    pub frag_total: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtrlFrame {
    pub kind: u8,
    pub src: DeviceId,
    pub dst: DeviceId,
    pub msg_id: u16,
    pub frag_idx: u16,
}

/// A decoded frame, dispatched by its leading type byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    Data { hdr: FragHeader, payload: Vec<u8> },
    Ctrl(CtrlFrame),
}

fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

impl FragHeader {
    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(FRAG_HDR_LEN + payload.len());
        out.push((self.kind & TYPE_MASK) | if self.ack_req { ACK_REQ } else { 0 });
        out.extend_from_slice(&self.src.to_be_bytes());
        out.extend_from_slice(&self.dst.to_be_bytes());
        out.extend_from_slice(&self.msg_id.to_be_bytes());
        out.extend_from_slice(&self.frag_idx.to_be_bytes());
        out.extend_from_slice(&self.frag_total.to_be_bytes());
        out.extend_from_slice(payload);
        out
    }
}

pub fn encode_hello(kind: u8, src: DeviceId, dst: DeviceId) -> Vec<u8> {
    let mut out = Vec::with_capacity(HELLO_HDR_LEN);
    out.push(kind);
    out.extend_from_slice(&src.to_be_bytes());
    out.extend_from_slice(&dst.to_be_bytes());
    out
}

pub fn encode_ack(src: DeviceId, dst: DeviceId, msg_id: u16, frag_idx: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(ACK_HDR_LEN);
    out.push(T_ACK);
    out.extend_from_slice(&src.to_be_bytes());
    out.extend_from_slice(&dst.to_be_bytes());
    out.extend_from_slice(&msg_id.to_be_bytes());
    out.extend_from_slice(&frag_idx.to_be_bytes());
    out
}

/// Parse one decoded packet payload. `None` means "malformed, ignore it" —
/// a frame that survived the CRC but is too short to be any known type.
pub fn parse(payload: &[u8]) -> Option<Frame> {
    let kind = *payload.first()?;

    if kind == T_HELLO || kind == T_HELLO_ACK {
        if payload.len() < HELLO_HDR_LEN {
            return None;
        }
        return Some(Frame::Ctrl(CtrlFrame {
            kind,
            src: be16(&payload[1..3]),
            dst: be16(&payload[3..5]),
            msg_id: 0,
            frag_idx: 0,
        }));
    }

    if kind == T_ACK {
        if payload.len() < ACK_HDR_LEN {
            return None;
        }
        return Some(Frame::Ctrl(CtrlFrame {
            kind,
            src: be16(&payload[1..3]),
            dst: be16(&payload[3..5]),
            msg_id: be16(&payload[5..7]),
            frag_idx: be16(&payload[7..9]),
        }));
    }

    if payload.len() < FRAG_HDR_LEN {
        return None;
    }
    let hdr = FragHeader {
        kind: kind & TYPE_MASK,
        ack_req: kind & ACK_REQ != 0,
        src: be16(&payload[1..3]),
        dst: be16(&payload[3..5]),
        msg_id: be16(&payload[5..7]),
        frag_idx: be16(&payload[7..9]),
        frag_total: be16(&payload[9..11]),
    };
    if hdr.frag_total == 0 || hdr.frag_idx >= hdr.frag_total {
        return None;
    }
    Some(Frame::Data { hdr, payload: payload[FRAG_HDR_LEN..].to_vec() })
}

// --------------------------------------------------------------------------- //
// File transfer metadata, prefixed to the first fragment of a T_FILE message:
//   name_len(1) name(name_len) size(4)
// --------------------------------------------------------------------------- //
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileMeta {
    pub name: String,
    pub size: u32,
}

pub fn encode_file_meta(name: &str, size: u32) -> Vec<u8> {
    let name = name.as_bytes();
    let n = name.len().min(255);
    let mut out = Vec::with_capacity(5 + n);
    out.push(n as u8);
    out.extend_from_slice(&name[..n]);
    out.extend_from_slice(&size.to_be_bytes());
    out
}

/// Split a reassembled T_FILE blob into its metadata and its contents.
pub fn decode_file_blob(blob: &[u8]) -> Option<(FileMeta, Vec<u8>)> {
    let nlen = *blob.first()? as usize;
    if blob.len() < 1 + nlen + 4 {
        return None;
    }
    let name = String::from_utf8_lossy(&blob[1..1 + nlen]).into_owned();
    let size = u32::from_be_bytes([
        blob[1 + nlen],
        blob[2 + nlen],
        blob[3 + nlen],
        blob[4 + nlen],
    ]);
    let start = 5 + nlen;
    let end = (start + size as usize).min(blob.len());
    Some((FileMeta { name, size }, blob[start..end].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_sizes_match_the_reference() {
        let h = FragHeader {
            kind: T_TEXT,
            ack_req: true,
            src: 0xA1A1,
            dst: 0xB2B2,
            msg_id: 7,
            frag_idx: 1,
            frag_total: 3,
        };
        assert_eq!(h.encode(&[]).len(), FRAG_HDR_LEN);
        assert_eq!(encode_hello(T_HELLO, 1, 2).len(), HELLO_HDR_LEN);
        assert_eq!(encode_ack(1, 2, 3, 4).len(), ACK_HDR_LEN);
    }

    #[test]
    fn data_frames_round_trip_with_the_ack_flag_intact() {
        for ack_req in [false, true] {
            let h = FragHeader {
                kind: T_FILE,
                ack_req,
                src: 0x1234,
                dst: 0xFEDC,
                msg_id: 0xBEEF,
                frag_idx: 2,
                frag_total: 9,
            };
            let wire = h.encode(b"body");
            // The flag lives in the high bit of the type byte.
            assert_eq!(wire[0] & ACK_REQ != 0, ack_req);
            assert_eq!(wire[0] & TYPE_MASK, T_FILE);
            match parse(&wire).unwrap() {
                Frame::Data { hdr, payload } => {
                    assert_eq!(hdr, h);
                    assert_eq!(payload, b"body");
                }
                other => panic!("expected data, got {other:?}"),
            }
        }
    }

    #[test]
    fn control_frames_round_trip() {
        match parse(&encode_ack(0xAAAA, 0xBBBB, 5, 6)).unwrap() {
            Frame::Ctrl(c) => {
                assert_eq!(
                    c,
                    CtrlFrame { kind: T_ACK, src: 0xAAAA, dst: 0xBBBB, msg_id: 5, frag_idx: 6 }
                );
            }
            other => panic!("expected ctrl, got {other:?}"),
        }
        match parse(&encode_hello(T_HELLO_ACK, 1, 2)).unwrap() {
            Frame::Ctrl(c) => assert_eq!(c.kind, T_HELLO_ACK),
            other => panic!("expected ctrl, got {other:?}"),
        }
    }

    #[test]
    fn malformed_frames_are_rejected_rather_than_panicking() {
        assert!(parse(&[]).is_none());
        assert!(parse(&[T_ACK, 1]).is_none(), "truncated ack");
        assert!(parse(&[T_TEXT, 0, 0]).is_none(), "truncated data header");
        // frag_total of zero, and an index past the end, are both nonsense.
        let mut h = FragHeader {
            kind: T_TEXT, ack_req: false, src: 1, dst: 2,
            msg_id: 3, frag_idx: 0, frag_total: 0,
        };
        assert!(parse(&h.encode(b"")).is_none());
        h.frag_total = 2;
        h.frag_idx = 2;
        assert!(parse(&h.encode(b"")).is_none());
    }

    #[test]
    fn file_metadata_round_trips() {
        let body = b"file contents here".to_vec();
        let mut blob = encode_file_meta("report.bin", body.len() as u32);
        blob.extend_from_slice(&body);
        let (meta, data) = decode_file_blob(&blob).unwrap();
        assert_eq!(meta, FileMeta { name: "report.bin".into(), size: body.len() as u32 });
        assert_eq!(data, body);
    }
}
