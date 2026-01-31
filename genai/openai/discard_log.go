package openai

import "github.com/kydenul/log"

var _ log.Logger = (*DiscardLog)(nil)

type DiscardLog struct{}

func NewDiscardLog() *DiscardLog { return &DiscardLog{} }

func (*DiscardLog) Sync() error               { return nil }
func (*DiscardLog) Debug(_ ...any)            {}
func (*DiscardLog) Debugf(_ string, _ ...any) {}
func (*DiscardLog) Debugw(_ string, _ ...any) {}
func (*DiscardLog) Debugln(_ ...any)          {}

func (*DiscardLog) Info(_ ...any)            {}
func (*DiscardLog) Infof(_ string, _ ...any) {}
func (*DiscardLog) Infow(_ string, _ ...any) {}
func (*DiscardLog) Infoln(_ ...any)          {}

func (*DiscardLog) Warn(_ ...any)            {}
func (*DiscardLog) Warnf(_ string, _ ...any) {}
func (*DiscardLog) Warnw(_ string, _ ...any) {}
func (*DiscardLog) Warnln(_ ...any)          {}

func (*DiscardLog) Error(_ ...any)            {}
func (*DiscardLog) Errorf(_ string, _ ...any) {}
func (*DiscardLog) Errorw(_ string, _ ...any) {}
func (*DiscardLog) Errorln(_ ...any)          {}

func (*DiscardLog) Panic(_ ...any)            {}
func (*DiscardLog) Panicf(_ string, _ ...any) {}
func (*DiscardLog) Panicw(_ string, _ ...any) {}
func (*DiscardLog) Panicln(_ ...any)          {}

func (*DiscardLog) Fatal(_ ...any)            {}
func (*DiscardLog) Fatalf(_ string, _ ...any) {}
func (*DiscardLog) Fatalw(_ string, _ ...any) {}
func (*DiscardLog) Fatalln(_ ...any)          {}
