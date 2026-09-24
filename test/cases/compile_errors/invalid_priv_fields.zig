const E = enum {
    a,
    priv b,
};
const T = struct {
    u8,
    priv u16,
};

// error
//
// :3:5: error: enum fields cannot be marked 'priv'
// :7:5: error: tuple fields cannot be marked 'priv'
