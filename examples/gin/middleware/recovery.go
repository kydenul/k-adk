package middleware

import (
	"runtime/debug"

	"github.com/gin-gonic/gin"
	"github.com/kydenul/log"
)

// Recovery returns a middleware that recovers from panics
func Recovery() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				log.Errorf("[GIN] panic recovered: %v\n%s", err, debug.Stack())
				c.AbortWithStatus(500)
			}
		}()
		c.Next()
	}
}
