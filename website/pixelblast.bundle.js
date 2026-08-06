(()=>{var n1=Object.create;var dv=Object.defineProperty;var i1=Object.getOwnPropertyDescriptor;var s1=Object.getOwnPropertyNames;var r1=Object.getPrototypeOf,a1=Object.prototype.hasOwnProperty;var Li=(t,e)=>()=>{try{return e||t((e={exports:{}}).exports,e),e.exports}catch(n){throw e=0,n}};var o1=(t,e,n,i)=>{if(e&&typeof e=="object"||typeof e=="function")for(let s of s1(e))!a1.call(t,s)&&s!==n&&dv(t,s,{get:()=>e[s],enumerable:!(i=i1(e,s))||i.enumerable});return t};var Tr=(t,e,n)=>(n=t!=null?n1(r1(t)):{},o1(e||!t||!t.__esModule?dv(n,"default",{value:t,enumerable:!0}):n,t));var Ev=Li(Le=>{"use strict";var Xh=Symbol.for("react.transitional.element"),l1=Symbol.for("react.portal"),c1=Symbol.for("react.fragment"),u1=Symbol.for("react.strict_mode"),f1=Symbol.for("react.profiler"),h1=Symbol.for("react.consumer"),d1=Symbol.for("react.context"),p1=Symbol.for("react.forward_ref"),m1=Symbol.for("react.suspense"),g1=Symbol.for("react.memo"),xv=Symbol.for("react.lazy"),v1=Symbol.for("react.activity"),pv=Symbol.iterator;function x1(t){return t===null||typeof t!="object"?null:(t=pv&&t[pv]||t["@@iterator"],typeof t=="function"?t:null)}var yv={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},_v=Object.assign,Sv={};function ua(t,e,n){this.props=t,this.context=e,this.refs=Sv,this.updater=n||yv}ua.prototype.isReactComponent={};ua.prototype.setState=function(t,e){if(typeof t!="object"&&typeof t!="function"&&t!=null)throw Error("takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,t,e,"setState")};ua.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this,t,"forceUpdate")};function Av(){}Av.prototype=ua.prototype;function Yh(t,e,n){this.props=t,this.context=e,this.refs=Sv,this.updater=n||yv}var qh=Yh.prototype=new Av;qh.constructor=Yh;_v(qh,ua.prototype);qh.isPureReactComponent=!0;var mv=Array.isArray;function Wh(){}var _t={H:null,A:null,T:null,S:null},Mv=Object.prototype.hasOwnProperty;function Qh(t,e,n){var i=n.ref;return{$$typeof:Xh,type:t,key:e,ref:i!==void 0?i:null,props:n}}function y1(t,e){return Qh(t.type,e,t.props)}function Zh(t){return typeof t=="object"&&t!==null&&t.$$typeof===Xh}function _1(t){var e={"=":"=0",":":"=2"};return"$"+t.replace(/[=:]/g,function(n){return e[n]})}var gv=/\/+/g;function kh(t,e){return typeof t=="object"&&t!==null&&t.key!=null?_1(""+t.key):e.toString(36)}function S1(t){switch(t.status){case"fulfilled":return t.value;case"rejected":throw t.reason;default:switch(typeof t.status=="string"?t.then(Wh,Wh):(t.status="pending",t.then(function(e){t.status==="pending"&&(t.status="fulfilled",t.value=e)},function(e){t.status==="pending"&&(t.status="rejected",t.reason=e)})),t.status){case"fulfilled":return t.value;case"rejected":throw t.reason}}throw t}function ca(t,e,n,i,s){var r=typeof t;(r==="undefined"||r==="boolean")&&(t=null);var a=!1;if(t===null)a=!0;else switch(r){case"bigint":case"string":case"number":a=!0;break;case"object":switch(t.$$typeof){case Xh:case l1:a=!0;break;case xv:return a=t._init,ca(a(t._payload),e,n,i,s)}}if(a)return s=s(t),a=i===""?"."+kh(t,0):i,mv(s)?(n="",a!=null&&(n=a.replace(gv,"$&/")+"/"),ca(s,e,n,"",function(c){return c})):s!=null&&(Zh(s)&&(s=y1(s,n+(s.key==null||t&&t.key===s.key?"":(""+s.key).replace(gv,"$&/")+"/")+a)),e.push(s)),1;a=0;var o=i===""?".":i+":";if(mv(t))for(var l=0;l<t.length;l++)i=t[l],r=o+kh(i,l),a+=ca(i,e,n,r,s);else if(l=x1(t),typeof l=="function")for(t=l.call(t),l=0;!(i=t.next()).done;)i=i.value,r=o+kh(i,l++),a+=ca(i,e,n,r,s);else if(r==="object"){if(typeof t.then=="function")return ca(S1(t),e,n,i,s);throw e=String(t),Error("Objects are not valid as a React child (found: "+(e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e)+"). If you meant to render a collection of children, use an array instead.")}return a}function fc(t,e,n){if(t==null)return t;var i=[],s=0;return ca(t,i,"","",function(r){return e.call(n,r,s++)}),i}function A1(t){if(t._status===-1){var e=t._result;e=e(),e.then(function(n){(t._status===0||t._status===-1)&&(t._status=1,t._result=n)},function(n){(t._status===0||t._status===-1)&&(t._status=2,t._result=n)}),t._status===-1&&(t._status=0,t._result=e)}if(t._status===1)return t._result.default;throw t._result}var vv=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)},M1={map:fc,forEach:function(t,e,n){fc(t,function(){e.apply(this,arguments)},n)},count:function(t){var e=0;return fc(t,function(){e++}),e},toArray:function(t){return fc(t,function(e){return e})||[]},only:function(t){if(!Zh(t))throw Error("React.Children.only expected to receive a single React element child.");return t}};Le.Activity=v1;Le.Children=M1;Le.Component=ua;Le.Fragment=c1;Le.Profiler=f1;Le.PureComponent=Yh;Le.StrictMode=u1;Le.Suspense=m1;Le.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=_t;Le.__COMPILER_RUNTIME={__proto__:null,c:function(t){return _t.H.useMemoCache(t)}};Le.cache=function(t){return function(){return t.apply(null,arguments)}};Le.cacheSignal=function(){return null};Le.cloneElement=function(t,e,n){if(t==null)throw Error("The argument must be a React element, but you passed "+t+".");var i=_v({},t.props),s=t.key;if(e!=null)for(r in e.key!==void 0&&(s=""+e.key),e)!Mv.call(e,r)||r==="key"||r==="__self"||r==="__source"||r==="ref"&&e.ref===void 0||(i[r]=e[r]);var r=arguments.length-2;if(r===1)i.children=n;else if(1<r){for(var a=Array(r),o=0;o<r;o++)a[o]=arguments[o+2];i.children=a}return Qh(t.type,s,i)};Le.createContext=function(t){return t={$$typeof:d1,_currentValue:t,_currentValue2:t,_threadCount:0,Provider:null,Consumer:null},t.Provider=t,t.Consumer={$$typeof:h1,_context:t},t};Le.createElement=function(t,e,n){var i,s={},r=null;if(e!=null)for(i in e.key!==void 0&&(r=""+e.key),e)Mv.call(e,i)&&i!=="key"&&i!=="__self"&&i!=="__source"&&(s[i]=e[i]);var a=arguments.length-2;if(a===1)s.children=n;else if(1<a){for(var o=Array(a),l=0;l<a;l++)o[l]=arguments[l+2];s.children=o}if(t&&t.defaultProps)for(i in a=t.defaultProps,a)s[i]===void 0&&(s[i]=a[i]);return Qh(t,r,s)};Le.createRef=function(){return{current:null}};Le.forwardRef=function(t){return{$$typeof:p1,render:t}};Le.isValidElement=Zh;Le.lazy=function(t){return{$$typeof:xv,_payload:{_status:-1,_result:t},_init:A1}};Le.memo=function(t,e){return{$$typeof:g1,type:t,compare:e===void 0?null:e}};Le.startTransition=function(t){var e=_t.T,n={};_t.T=n;try{var i=t(),s=_t.S;s!==null&&s(n,i),typeof i=="object"&&i!==null&&typeof i.then=="function"&&i.then(Wh,vv)}catch(r){vv(r)}finally{e!==null&&n.types!==null&&(e.types=n.types),_t.T=e}};Le.unstable_useCacheRefresh=function(){return _t.H.useCacheRefresh()};Le.use=function(t){return _t.H.use(t)};Le.useActionState=function(t,e,n){return _t.H.useActionState(t,e,n)};Le.useCallback=function(t,e){return _t.H.useCallback(t,e)};Le.useContext=function(t){return _t.H.useContext(t)};Le.useDebugValue=function(){};Le.useDeferredValue=function(t,e){return _t.H.useDeferredValue(t,e)};Le.useEffect=function(t,e){return _t.H.useEffect(t,e)};Le.useEffectEvent=function(t){return _t.H.useEffectEvent(t)};Le.useId=function(){return _t.H.useId()};Le.useImperativeHandle=function(t,e,n){return _t.H.useImperativeHandle(t,e,n)};Le.useInsertionEffect=function(t,e){return _t.H.useInsertionEffect(t,e)};Le.useLayoutEffect=function(t,e){return _t.H.useLayoutEffect(t,e)};Le.useMemo=function(t,e){return _t.H.useMemo(t,e)};Le.useOptimistic=function(t,e){return _t.H.useOptimistic(t,e)};Le.useReducer=function(t,e,n){return _t.H.useReducer(t,e,n)};Le.useRef=function(t){return _t.H.useRef(t)};Le.useState=function(t){return _t.H.useState(t)};Le.useSyncExternalStore=function(t,e,n){return _t.H.useSyncExternalStore(t,e,n)};Le.useTransition=function(){return _t.H.useTransition()};Le.version="19.2.8"});var bo=Li((ED,Tv)=>{"use strict";Tv.exports=Ev()});var Lv=Li(bt=>{"use strict";function $h(t,e){var n=t.length;t.push(e);e:for(;0<n;){var i=n-1>>>1,s=t[i];if(0<hc(s,e))t[i]=e,t[n]=s,n=i;else break e}}function Ni(t){return t.length===0?null:t[0]}function pc(t){if(t.length===0)return null;var e=t[0],n=t.pop();if(n!==e){t[0]=n;e:for(var i=0,s=t.length,r=s>>>1;i<r;){var a=2*(i+1)-1,o=t[a],l=a+1,c=t[l];if(0>hc(o,n))l<s&&0>hc(c,o)?(t[i]=c,t[l]=n,i=l):(t[i]=o,t[a]=n,i=a);else if(l<s&&0>hc(c,n))t[i]=c,t[l]=n,i=l;else break e}}return e}function hc(t,e){var n=t.sortIndex-e.sortIndex;return n!==0?n:t.id-e.id}bt.unstable_now=void 0;typeof performance=="object"&&typeof performance.now=="function"?(bv=performance,bt.unstable_now=function(){return bv.now()}):(Kh=Date,wv=Kh.now(),bt.unstable_now=function(){return Kh.now()-wv});var bv,Kh,wv,$i=[],Ds=[],E1=1,si=null,xn=3,ed=!1,wo=!1,Co=!1,td=!1,Dv=typeof setTimeout=="function"?setTimeout:null,Uv=typeof clearTimeout=="function"?clearTimeout:null,Cv=typeof setImmediate<"u"?setImmediate:null;function dc(t){for(var e=Ni(Ds);e!==null;){if(e.callback===null)pc(Ds);else if(e.startTime<=t)pc(Ds),e.sortIndex=e.expirationTime,$h($i,e);else break;e=Ni(Ds)}}function nd(t){if(Co=!1,dc(t),!wo)if(Ni($i)!==null)wo=!0,ha||(ha=!0,fa());else{var e=Ni(Ds);e!==null&&id(nd,e.startTime-t)}}var ha=!1,Ro=-1,Bv=5,Iv=-1;function Pv(){return td?!0:!(bt.unstable_now()-Iv<Bv)}function Jh(){if(td=!1,ha){var t=bt.unstable_now();Iv=t;var e=!0;try{e:{wo=!1,Co&&(Co=!1,Uv(Ro),Ro=-1),ed=!0;var n=xn;try{t:{for(dc(t),si=Ni($i);si!==null&&!(si.expirationTime>t&&Pv());){var i=si.callback;if(typeof i=="function"){si.callback=null,xn=si.priorityLevel;var s=i(si.expirationTime<=t);if(t=bt.unstable_now(),typeof s=="function"){si.callback=s,dc(t),e=!0;break t}si===Ni($i)&&pc($i),dc(t)}else pc($i);si=Ni($i)}if(si!==null)e=!0;else{var r=Ni(Ds);r!==null&&id(nd,r.startTime-t),e=!1}}break e}finally{si=null,xn=n,ed=!1}e=void 0}}finally{e?fa():ha=!1}}}var fa;typeof Cv=="function"?fa=function(){Cv(Jh)}:typeof MessageChannel<"u"?(jh=new MessageChannel,Rv=jh.port2,jh.port1.onmessage=Jh,fa=function(){Rv.postMessage(null)}):fa=function(){Dv(Jh,0)};var jh,Rv;function id(t,e){Ro=Dv(function(){t(bt.unstable_now())},e)}bt.unstable_IdlePriority=5;bt.unstable_ImmediatePriority=1;bt.unstable_LowPriority=4;bt.unstable_NormalPriority=3;bt.unstable_Profiling=null;bt.unstable_UserBlockingPriority=2;bt.unstable_cancelCallback=function(t){t.callback=null};bt.unstable_forceFrameRate=function(t){0>t||125<t?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):Bv=0<t?Math.floor(1e3/t):5};bt.unstable_getCurrentPriorityLevel=function(){return xn};bt.unstable_next=function(t){switch(xn){case 1:case 2:case 3:var e=3;break;default:e=xn}var n=xn;xn=e;try{return t()}finally{xn=n}};bt.unstable_requestPaint=function(){td=!0};bt.unstable_runWithPriority=function(t,e){switch(t){case 1:case 2:case 3:case 4:case 5:break;default:t=3}var n=xn;xn=t;try{return e()}finally{xn=n}};bt.unstable_scheduleCallback=function(t,e,n){var i=bt.unstable_now();switch(typeof n=="object"&&n!==null?(n=n.delay,n=typeof n=="number"&&0<n?i+n:i):n=i,t){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=n+s,t={id:E1++,callback:e,priorityLevel:t,startTime:n,expirationTime:s,sortIndex:-1},n>i?(t.sortIndex=n,$h(Ds,t),Ni($i)===null&&t===Ni(Ds)&&(Co?(Uv(Ro),Ro=-1):Co=!0,id(nd,n-i))):(t.sortIndex=s,$h($i,t),wo||ed||(wo=!0,ha||(ha=!0,fa()))),t};bt.unstable_shouldYield=Pv;bt.unstable_wrapCallback=function(t){var e=xn;return function(){var n=xn;xn=e;try{return t.apply(this,arguments)}finally{xn=n}}}});var Ov=Li((bD,Nv)=>{"use strict";Nv.exports=Lv()});var zv=Li(wn=>{"use strict";var T1=bo();function Fv(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function Us(){}var bn={d:{f:Us,r:function(){throw Error(Fv(522))},D:Us,C:Us,L:Us,m:Us,X:Us,S:Us,M:Us},p:0,findDOMNode:null},b1=Symbol.for("react.portal");function w1(t,e,n){var i=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:b1,key:i==null?null:""+i,children:t,containerInfo:e,implementation:n}}var Do=T1.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function mc(t,e){if(t==="font")return"";if(typeof e=="string")return e==="use-credentials"?e:""}wn.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=bn;wn.createPortal=function(t,e){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)throw Error(Fv(299));return w1(t,e,null,n)};wn.flushSync=function(t){var e=Do.T,n=bn.p;try{if(Do.T=null,bn.p=2,t)return t()}finally{Do.T=e,bn.p=n,bn.d.f()}};wn.preconnect=function(t,e){typeof t=="string"&&(e?(e=e.crossOrigin,e=typeof e=="string"?e==="use-credentials"?e:"":void 0):e=null,bn.d.C(t,e))};wn.prefetchDNS=function(t){typeof t=="string"&&bn.d.D(t)};wn.preinit=function(t,e){if(typeof t=="string"&&e&&typeof e.as=="string"){var n=e.as,i=mc(n,e.crossOrigin),s=typeof e.integrity=="string"?e.integrity:void 0,r=typeof e.fetchPriority=="string"?e.fetchPriority:void 0;n==="style"?bn.d.S(t,typeof e.precedence=="string"?e.precedence:void 0,{crossOrigin:i,integrity:s,fetchPriority:r}):n==="script"&&bn.d.X(t,{crossOrigin:i,integrity:s,fetchPriority:r,nonce:typeof e.nonce=="string"?e.nonce:void 0})}};wn.preinitModule=function(t,e){if(typeof t=="string")if(typeof e=="object"&&e!==null){if(e.as==null||e.as==="script"){var n=mc(e.as,e.crossOrigin);bn.d.M(t,{crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0})}}else e==null&&bn.d.M(t)};wn.preload=function(t,e){if(typeof t=="string"&&typeof e=="object"&&e!==null&&typeof e.as=="string"){var n=e.as,i=mc(n,e.crossOrigin);bn.d.L(t,n,{crossOrigin:i,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,type:typeof e.type=="string"?e.type:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0,referrerPolicy:typeof e.referrerPolicy=="string"?e.referrerPolicy:void 0,imageSrcSet:typeof e.imageSrcSet=="string"?e.imageSrcSet:void 0,imageSizes:typeof e.imageSizes=="string"?e.imageSizes:void 0,media:typeof e.media=="string"?e.media:void 0})}};wn.preloadModule=function(t,e){if(typeof t=="string")if(e){var n=mc(e.as,e.crossOrigin);bn.d.m(t,{as:typeof e.as=="string"&&e.as!=="script"?e.as:void 0,crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0})}else bn.d.m(t)};wn.requestFormReset=function(t){bn.d.r(t)};wn.unstable_batchedUpdates=function(t,e){return t(e)};wn.useFormState=function(t,e,n){return Do.H.useFormState(t,e,n)};wn.useFormStatus=function(){return Do.H.useHostTransitionStatus()};wn.version="19.2.8"});var Vv=Li((CD,Hv)=>{"use strict";function Gv(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Gv)}catch(t){console.error(t)}}Gv(),Hv.exports=zv()});var eS=Li(Gu=>{"use strict";var tn=Ov(),px=bo(),C1=Vv();function J(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function mx(t){return!(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)}function gl(t){var e=t,n=t;if(t.alternate)for(;e.return;)e=e.return;else{t=e;do e=t,(e.flags&4098)!==0&&(n=e.return),t=e.return;while(t)}return e.tag===3?n:null}function gx(t){if(t.tag===13){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function vx(t){if(t.tag===31){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function kv(t){if(gl(t)!==t)throw Error(J(188))}function R1(t){var e=t.alternate;if(!e){if(e=gl(t),e===null)throw Error(J(188));return e!==t?null:t}for(var n=t,i=e;;){var s=n.return;if(s===null)break;var r=s.alternate;if(r===null){if(i=s.return,i!==null){n=i;continue}break}if(s.child===r.child){for(r=s.child;r;){if(r===n)return kv(s),t;if(r===i)return kv(s),e;r=r.sibling}throw Error(J(188))}if(n.return!==i.return)n=s,i=r;else{for(var a=!1,o=s.child;o;){if(o===n){a=!0,n=s,i=r;break}if(o===i){a=!0,i=s,n=r;break}o=o.sibling}if(!a){for(o=r.child;o;){if(o===n){a=!0,n=r,i=s;break}if(o===i){a=!0,i=r,n=s;break}o=o.sibling}if(!a)throw Error(J(189))}}if(n.alternate!==i)throw Error(J(190))}if(n.tag!==3)throw Error(J(188));return n.stateNode.current===n?t:e}function xx(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t;for(t=t.child;t!==null;){if(e=xx(t),e!==null)return e;t=t.sibling}return null}var Mt=Object.assign,D1=Symbol.for("react.element"),gc=Symbol.for("react.transitional.element"),Fo=Symbol.for("react.portal"),xa=Symbol.for("react.fragment"),yx=Symbol.for("react.strict_mode"),Od=Symbol.for("react.profiler"),_x=Symbol.for("react.consumer"),os=Symbol.for("react.context"),Bp=Symbol.for("react.forward_ref"),Fd=Symbol.for("react.suspense"),zd=Symbol.for("react.suspense_list"),Ip=Symbol.for("react.memo"),Bs=Symbol.for("react.lazy"),Gd=Symbol.for("react.activity"),U1=Symbol.for("react.memo_cache_sentinel"),Wv=Symbol.iterator;function Uo(t){return t===null||typeof t!="object"?null:(t=Wv&&t[Wv]||t["@@iterator"],typeof t=="function"?t:null)}var B1=Symbol.for("react.client.reference");function Hd(t){if(t==null)return null;if(typeof t=="function")return t.$$typeof===B1?null:t.displayName||t.name||null;if(typeof t=="string")return t;switch(t){case xa:return"Fragment";case Od:return"Profiler";case yx:return"StrictMode";case Fd:return"Suspense";case zd:return"SuspenseList";case Gd:return"Activity"}if(typeof t=="object")switch(t.$$typeof){case Fo:return"Portal";case os:return t.displayName||"Context";case _x:return(t._context.displayName||"Context")+".Consumer";case Bp:var e=t.render;return t=t.displayName,t||(t=e.displayName||e.name||"",t=t!==""?"ForwardRef("+t+")":"ForwardRef"),t;case Ip:return e=t.displayName||null,e!==null?e:Hd(t.type)||"Memo";case Bs:e=t._payload,t=t._init;try{return Hd(t(e))}catch{}}return null}var zo=Array.isArray,Be=px.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,it=C1.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,Ur={pending:!1,data:null,method:null,action:null},Vd=[],ya=-1;function Hi(t){return{current:t}}function ln(t){0>ya||(t.current=Vd[ya],Vd[ya]=null,ya--)}function vt(t,e){ya++,Vd[ya]=t.current,t.current=e}var Gi=Hi(null),nl=Hi(null),ks=Hi(null),Zc=Hi(null);function Kc(t,e){switch(vt(ks,e),vt(nl,t),vt(Gi,null),e.nodeType){case 9:case 11:t=(t=e.documentElement)&&(t=t.namespaceURI)?J0(t):0;break;default:if(t=e.tagName,e=e.namespaceURI)e=J0(e),t=G_(e,t);else switch(t){case"svg":t=1;break;case"math":t=2;break;default:t=0}}ln(Gi),vt(Gi,t)}function Oa(){ln(Gi),ln(nl),ln(ks)}function kd(t){t.memoizedState!==null&&vt(Zc,t);var e=Gi.current,n=G_(e,t.type);e!==n&&(vt(nl,t),vt(Gi,n))}function Jc(t){nl.current===t&&(ln(Gi),ln(nl)),Zc.current===t&&(ln(Zc),dl._currentValue=Ur)}var sd,Xv;function wr(t){if(sd===void 0)try{throw Error()}catch(n){var e=n.stack.trim().match(/\n( *(at )?)/);sd=e&&e[1]||"",Xv=-1<n.stack.indexOf(`
    at`)?" (<anonymous>)":-1<n.stack.indexOf("@")?"@unknown:0:0":""}return`
`+sd+t+Xv}var rd=!1;function ad(t,e){if(!t||rd)return"";rd=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var i={DetermineComponentFrameRoot:function(){try{if(e){var p=function(){throw Error()};if(Object.defineProperty(p.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(p,[])}catch(d){var u=d}Reflect.construct(t,[],p)}else{try{p.call()}catch(d){u=d}t.call(p.prototype)}}else{try{throw Error()}catch(d){u=d}(p=t())&&typeof p.catch=="function"&&p.catch(function(){})}}catch(d){if(d&&u&&typeof d.stack=="string")return[d.stack,u.stack]}return[null,null]}};i.DetermineComponentFrameRoot.displayName="DetermineComponentFrameRoot";var s=Object.getOwnPropertyDescriptor(i.DetermineComponentFrameRoot,"name");s&&s.configurable&&Object.defineProperty(i.DetermineComponentFrameRoot,"name",{value:"DetermineComponentFrameRoot"});var r=i.DetermineComponentFrameRoot(),a=r[0],o=r[1];if(a&&o){var l=a.split(`
`),c=o.split(`
`);for(s=i=0;i<l.length&&!l[i].includes("DetermineComponentFrameRoot");)i++;for(;s<c.length&&!c[s].includes("DetermineComponentFrameRoot");)s++;if(i===l.length||s===c.length)for(i=l.length-1,s=c.length-1;1<=i&&0<=s&&l[i]!==c[s];)s--;for(;1<=i&&0<=s;i--,s--)if(l[i]!==c[s]){if(i!==1||s!==1)do if(i--,s--,0>s||l[i]!==c[s]){var h=`
`+l[i].replace(" at new "," at ");return t.displayName&&h.includes("<anonymous>")&&(h=h.replace("<anonymous>",t.displayName)),h}while(1<=i&&0<=s);break}}}finally{rd=!1,Error.prepareStackTrace=n}return(n=t?t.displayName||t.name:"")?wr(n):""}function I1(t,e){switch(t.tag){case 26:case 27:case 5:return wr(t.type);case 16:return wr("Lazy");case 13:return t.child!==e&&e!==null?wr("Suspense Fallback"):wr("Suspense");case 19:return wr("SuspenseList");case 0:case 15:return ad(t.type,!1);case 11:return ad(t.type.render,!1);case 1:return ad(t.type,!0);case 31:return wr("Activity");default:return""}}function Yv(t){try{var e="",n=null;do e+=I1(t,n),n=t,t=t.return;while(t);return e}catch(i){return`
Error generating stack: `+i.message+`
`+i.stack}}var Wd=Object.prototype.hasOwnProperty,Pp=tn.unstable_scheduleCallback,od=tn.unstable_cancelCallback,P1=tn.unstable_shouldYield,L1=tn.unstable_requestPaint,Yn=tn.unstable_now,N1=tn.unstable_getCurrentPriorityLevel,Sx=tn.unstable_ImmediatePriority,Ax=tn.unstable_UserBlockingPriority,jc=tn.unstable_NormalPriority,O1=tn.unstable_LowPriority,Mx=tn.unstable_IdlePriority,F1=tn.log,z1=tn.unstable_setDisableYieldValue,vl=null,qn=null;function Fs(t){if(typeof F1=="function"&&z1(t),qn&&typeof qn.setStrictMode=="function")try{qn.setStrictMode(vl,t)}catch{}}var Qn=Math.clz32?Math.clz32:V1,G1=Math.log,H1=Math.LN2;function V1(t){return t>>>=0,t===0?32:31-(G1(t)/H1|0)|0}var vc=256,xc=262144,yc=4194304;function Cr(t){var e=t&42;if(e!==0)return e;switch(t&-t){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return t&261888;case 262144:case 524288:case 1048576:case 2097152:return t&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return t&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return t}}function Eu(t,e,n){var i=t.pendingLanes;if(i===0)return 0;var s=0,r=t.suspendedLanes,a=t.pingedLanes;t=t.warmLanes;var o=i&134217727;return o!==0?(i=o&~r,i!==0?s=Cr(i):(a&=o,a!==0?s=Cr(a):n||(n=o&~t,n!==0&&(s=Cr(n))))):(o=i&~r,o!==0?s=Cr(o):a!==0?s=Cr(a):n||(n=i&~t,n!==0&&(s=Cr(n)))),s===0?0:e!==0&&e!==s&&(e&r)===0&&(r=s&-s,n=e&-e,r>=n||r===32&&(n&4194048)!==0)?e:s}function xl(t,e){return(t.pendingLanes&~(t.suspendedLanes&~t.pingedLanes)&e)===0}function k1(t,e){switch(t){case 1:case 2:case 4:case 8:case 64:return e+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Ex(){var t=yc;return yc<<=1,(yc&62914560)===0&&(yc=4194304),t}function ld(t){for(var e=[],n=0;31>n;n++)e.push(t);return e}function yl(t,e){t.pendingLanes|=e,e!==268435456&&(t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0)}function W1(t,e,n,i,s,r){var a=t.pendingLanes;t.pendingLanes=n,t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0,t.expiredLanes&=n,t.entangledLanes&=n,t.errorRecoveryDisabledLanes&=n,t.shellSuspendCounter=0;var o=t.entanglements,l=t.expirationTimes,c=t.hiddenUpdates;for(n=a&~n;0<n;){var h=31-Qn(n),p=1<<h;o[h]=0,l[h]=-1;var u=c[h];if(u!==null)for(c[h]=null,h=0;h<u.length;h++){var d=u[h];d!==null&&(d.lane&=-536870913)}n&=~p}i!==0&&Tx(t,i,0),r!==0&&s===0&&t.tag!==0&&(t.suspendedLanes|=r&~(a&~e))}function Tx(t,e,n){t.pendingLanes|=e,t.suspendedLanes&=~e;var i=31-Qn(e);t.entangledLanes|=e,t.entanglements[i]=t.entanglements[i]|1073741824|n&261930}function bx(t,e){var n=t.entangledLanes|=e;for(t=t.entanglements;n;){var i=31-Qn(n),s=1<<i;s&e|t[i]&e&&(t[i]|=e),n&=~s}}function wx(t,e){var n=e&-e;return n=(n&42)!==0?1:Lp(n),(n&(t.suspendedLanes|e))!==0?0:n}function Lp(t){switch(t){case 2:t=1;break;case 8:t=4;break;case 32:t=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:t=128;break;case 268435456:t=134217728;break;default:t=0}return t}function Np(t){return t&=-t,2<t?8<t?(t&134217727)!==0?32:268435456:8:2}function Cx(){var t=it.p;return t!==0?t:(t=window.event,t===void 0?32:J_(t.type))}function qv(t,e){var n=it.p;try{return it.p=t,e()}finally{it.p=n}}var nr=Math.random().toString(36).slice(2),dn="__reactFiber$"+nr,Nn="__reactProps$"+nr,Qa="__reactContainer$"+nr,Xd="__reactEvents$"+nr,X1="__reactListeners$"+nr,Y1="__reactHandles$"+nr,Qv="__reactResources$"+nr,_l="__reactMarker$"+nr;function Op(t){delete t[dn],delete t[Nn],delete t[Xd],delete t[X1],delete t[Y1]}function _a(t){var e=t[dn];if(e)return e;for(var n=t.parentNode;n;){if(e=n[Qa]||n[dn]){if(n=e.alternate,e.child!==null||n!==null&&n.child!==null)for(t=nx(t);t!==null;){if(n=t[dn])return n;t=nx(t)}return e}t=n,n=t.parentNode}return null}function Za(t){if(t=t[dn]||t[Qa]){var e=t.tag;if(e===5||e===6||e===13||e===31||e===26||e===27||e===3)return t}return null}function Go(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t.stateNode;throw Error(J(33))}function Da(t){var e=t[Qv];return e||(e=t[Qv]={hoistableStyles:new Map,hoistableScripts:new Map}),e}function on(t){t[_l]=!0}var Rx=new Set,Dx={};function Hr(t,e){Fa(t,e),Fa(t+"Capture",e)}function Fa(t,e){for(Dx[t]=e,t=0;t<e.length;t++)Rx.add(e[t])}var q1=RegExp("^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"),Zv={},Kv={};function Q1(t){return Wd.call(Kv,t)?!0:Wd.call(Zv,t)?!1:q1.test(t)?Kv[t]=!0:(Zv[t]=!0,!1)}function Pc(t,e,n){if(Q1(e))if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":t.removeAttribute(e);return;case"boolean":var i=e.toLowerCase().slice(0,5);if(i!=="data-"&&i!=="aria-"){t.removeAttribute(e);return}}t.setAttribute(e,""+n)}}function _c(t,e,n){if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(e);return}t.setAttribute(e,""+n)}}function es(t,e,n,i){if(i===null)t.removeAttribute(n);else{switch(typeof i){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(n);return}t.setAttributeNS(e,n,""+i)}}function ai(t){switch(typeof t){case"bigint":case"boolean":case"number":case"string":case"undefined":return t;case"object":return t;default:return""}}function Ux(t){var e=t.type;return(t=t.nodeName)&&t.toLowerCase()==="input"&&(e==="checkbox"||e==="radio")}function Z1(t,e,n){var i=Object.getOwnPropertyDescriptor(t.constructor.prototype,e);if(!t.hasOwnProperty(e)&&typeof i<"u"&&typeof i.get=="function"&&typeof i.set=="function"){var s=i.get,r=i.set;return Object.defineProperty(t,e,{configurable:!0,get:function(){return s.call(this)},set:function(a){n=""+a,r.call(this,a)}}),Object.defineProperty(t,e,{enumerable:i.enumerable}),{getValue:function(){return n},setValue:function(a){n=""+a},stopTracking:function(){t._valueTracker=null,delete t[e]}}}}function Yd(t){if(!t._valueTracker){var e=Ux(t)?"checked":"value";t._valueTracker=Z1(t,e,""+t[e])}}function Bx(t){if(!t)return!1;var e=t._valueTracker;if(!e)return!0;var n=e.getValue(),i="";return t&&(i=Ux(t)?t.checked?"true":"false":t.value),t=i,t!==n?(e.setValue(t),!0):!1}function $c(t){if(t=t||(typeof document<"u"?document:void 0),typeof t>"u")return null;try{return t.activeElement||t.body}catch{return t.body}}var K1=/[\n"\\]/g;function ci(t){return t.replace(K1,function(e){return"\\"+e.charCodeAt(0).toString(16)+" "})}function qd(t,e,n,i,s,r,a,o){t.name="",a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"?t.type=a:t.removeAttribute("type"),e!=null?a==="number"?(e===0&&t.value===""||t.value!=e)&&(t.value=""+ai(e)):t.value!==""+ai(e)&&(t.value=""+ai(e)):a!=="submit"&&a!=="reset"||t.removeAttribute("value"),e!=null?Qd(t,a,ai(e)):n!=null?Qd(t,a,ai(n)):i!=null&&t.removeAttribute("value"),s==null&&r!=null&&(t.defaultChecked=!!r),s!=null&&(t.checked=s&&typeof s!="function"&&typeof s!="symbol"),o!=null&&typeof o!="function"&&typeof o!="symbol"&&typeof o!="boolean"?t.name=""+ai(o):t.removeAttribute("name")}function Ix(t,e,n,i,s,r,a,o){if(r!=null&&typeof r!="function"&&typeof r!="symbol"&&typeof r!="boolean"&&(t.type=r),e!=null||n!=null){if(!(r!=="submit"&&r!=="reset"||e!=null)){Yd(t);return}n=n!=null?""+ai(n):"",e=e!=null?""+ai(e):n,o||e===t.value||(t.value=e),t.defaultValue=e}i=i??s,i=typeof i!="function"&&typeof i!="symbol"&&!!i,t.checked=o?t.checked:!!i,t.defaultChecked=!!i,a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"&&(t.name=a),Yd(t)}function Qd(t,e,n){e==="number"&&$c(t.ownerDocument)===t||t.defaultValue===""+n||(t.defaultValue=""+n)}function Ua(t,e,n,i){if(t=t.options,e){e={};for(var s=0;s<n.length;s++)e["$"+n[s]]=!0;for(n=0;n<t.length;n++)s=e.hasOwnProperty("$"+t[n].value),t[n].selected!==s&&(t[n].selected=s),s&&i&&(t[n].defaultSelected=!0)}else{for(n=""+ai(n),e=null,s=0;s<t.length;s++){if(t[s].value===n){t[s].selected=!0,i&&(t[s].defaultSelected=!0);return}e!==null||t[s].disabled||(e=t[s])}e!==null&&(e.selected=!0)}}function Px(t,e,n){if(e!=null&&(e=""+ai(e),e!==t.value&&(t.value=e),n==null)){t.defaultValue!==e&&(t.defaultValue=e);return}t.defaultValue=n!=null?""+ai(n):""}function Lx(t,e,n,i){if(e==null){if(i!=null){if(n!=null)throw Error(J(92));if(zo(i)){if(1<i.length)throw Error(J(93));i=i[0]}n=i}n==null&&(n=""),e=n}n=ai(e),t.defaultValue=n,i=t.textContent,i===n&&i!==""&&i!==null&&(t.value=i),Yd(t)}function za(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&n.nodeType===3){n.nodeValue=e;return}}t.textContent=e}var J1=new Set("animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(" "));function Jv(t,e,n){var i=e.indexOf("--")===0;n==null||typeof n=="boolean"||n===""?i?t.setProperty(e,""):e==="float"?t.cssFloat="":t[e]="":i?t.setProperty(e,n):typeof n!="number"||n===0||J1.has(e)?e==="float"?t.cssFloat=n:t[e]=(""+n).trim():t[e]=n+"px"}function Nx(t,e,n){if(e!=null&&typeof e!="object")throw Error(J(62));if(t=t.style,n!=null){for(var i in n)!n.hasOwnProperty(i)||e!=null&&e.hasOwnProperty(i)||(i.indexOf("--")===0?t.setProperty(i,""):i==="float"?t.cssFloat="":t[i]="");for(var s in e)i=e[s],e.hasOwnProperty(s)&&n[s]!==i&&Jv(t,s,i)}else for(var r in e)e.hasOwnProperty(r)&&Jv(t,r,e[r])}function Fp(t){if(t.indexOf("-")===-1)return!1;switch(t){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var j1=new Map([["acceptCharset","accept-charset"],["htmlFor","for"],["httpEquiv","http-equiv"],["crossOrigin","crossorigin"],["accentHeight","accent-height"],["alignmentBaseline","alignment-baseline"],["arabicForm","arabic-form"],["baselineShift","baseline-shift"],["capHeight","cap-height"],["clipPath","clip-path"],["clipRule","clip-rule"],["colorInterpolation","color-interpolation"],["colorInterpolationFilters","color-interpolation-filters"],["colorProfile","color-profile"],["colorRendering","color-rendering"],["dominantBaseline","dominant-baseline"],["enableBackground","enable-background"],["fillOpacity","fill-opacity"],["fillRule","fill-rule"],["floodColor","flood-color"],["floodOpacity","flood-opacity"],["fontFamily","font-family"],["fontSize","font-size"],["fontSizeAdjust","font-size-adjust"],["fontStretch","font-stretch"],["fontStyle","font-style"],["fontVariant","font-variant"],["fontWeight","font-weight"],["glyphName","glyph-name"],["glyphOrientationHorizontal","glyph-orientation-horizontal"],["glyphOrientationVertical","glyph-orientation-vertical"],["horizAdvX","horiz-adv-x"],["horizOriginX","horiz-origin-x"],["imageRendering","image-rendering"],["letterSpacing","letter-spacing"],["lightingColor","lighting-color"],["markerEnd","marker-end"],["markerMid","marker-mid"],["markerStart","marker-start"],["overlinePosition","overline-position"],["overlineThickness","overline-thickness"],["paintOrder","paint-order"],["panose-1","panose-1"],["pointerEvents","pointer-events"],["renderingIntent","rendering-intent"],["shapeRendering","shape-rendering"],["stopColor","stop-color"],["stopOpacity","stop-opacity"],["strikethroughPosition","strikethrough-position"],["strikethroughThickness","strikethrough-thickness"],["strokeDasharray","stroke-dasharray"],["strokeDashoffset","stroke-dashoffset"],["strokeLinecap","stroke-linecap"],["strokeLinejoin","stroke-linejoin"],["strokeMiterlimit","stroke-miterlimit"],["strokeOpacity","stroke-opacity"],["strokeWidth","stroke-width"],["textAnchor","text-anchor"],["textDecoration","text-decoration"],["textRendering","text-rendering"],["transformOrigin","transform-origin"],["underlinePosition","underline-position"],["underlineThickness","underline-thickness"],["unicodeBidi","unicode-bidi"],["unicodeRange","unicode-range"],["unitsPerEm","units-per-em"],["vAlphabetic","v-alphabetic"],["vHanging","v-hanging"],["vIdeographic","v-ideographic"],["vMathematical","v-mathematical"],["vectorEffect","vector-effect"],["vertAdvY","vert-adv-y"],["vertOriginX","vert-origin-x"],["vertOriginY","vert-origin-y"],["wordSpacing","word-spacing"],["writingMode","writing-mode"],["xmlnsXlink","xmlns:xlink"],["xHeight","x-height"]]),$1=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function Lc(t){return $1.test(""+t)?"javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')":t}function ls(){}var Zd=null;function zp(t){return t=t.target||t.srcElement||window,t.correspondingUseElement&&(t=t.correspondingUseElement),t.nodeType===3?t.parentNode:t}var Sa=null,Ba=null;function jv(t){var e=Za(t);if(e&&(t=e.stateNode)){var n=t[Nn]||null;e:switch(t=e.stateNode,e.type){case"input":if(qd(t,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),e=n.name,n.type==="radio"&&e!=null){for(n=t;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll('input[name="'+ci(""+e)+'"][type="radio"]'),e=0;e<n.length;e++){var i=n[e];if(i!==t&&i.form===t.form){var s=i[Nn]||null;if(!s)throw Error(J(90));qd(i,s.value,s.defaultValue,s.defaultValue,s.checked,s.defaultChecked,s.type,s.name)}}for(e=0;e<n.length;e++)i=n[e],i.form===t.form&&Bx(i)}break e;case"textarea":Px(t,n.value,n.defaultValue);break e;case"select":e=n.value,e!=null&&Ua(t,!!n.multiple,e,!1)}}}var cd=!1;function Ox(t,e,n){if(cd)return t(e,n);cd=!0;try{var i=t(e);return i}finally{if(cd=!1,(Sa!==null||Ba!==null)&&(Nu(),Sa&&(e=Sa,t=Ba,Ba=Sa=null,jv(e),t)))for(e=0;e<t.length;e++)jv(t[e])}}function il(t,e){var n=t.stateNode;if(n===null)return null;var i=n[Nn]||null;if(i===null)return null;n=i[e];e:switch(e){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(i=!i.disabled)||(t=t.type,i=!(t==="button"||t==="input"||t==="select"||t==="textarea")),t=!i;break e;default:t=!1}if(t)return null;if(n&&typeof n!="function")throw Error(J(231,e,typeof n));return n}var ds=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Kd=!1;if(ds)try{da={},Object.defineProperty(da,"passive",{get:function(){Kd=!0}}),window.addEventListener("test",da,da),window.removeEventListener("test",da,da)}catch{Kd=!1}var da,zs=null,Gp=null,Nc=null;function Fx(){if(Nc)return Nc;var t,e=Gp,n=e.length,i,s="value"in zs?zs.value:zs.textContent,r=s.length;for(t=0;t<n&&e[t]===s[t];t++);var a=n-t;for(i=1;i<=a&&e[n-i]===s[r-i];i++);return Nc=s.slice(t,1<i?1-i:void 0)}function Oc(t){var e=t.keyCode;return"charCode"in t?(t=t.charCode,t===0&&e===13&&(t=13)):t=e,t===10&&(t=13),32<=t||t===13?t:0}function Sc(){return!0}function $v(){return!1}function On(t){function e(n,i,s,r,a){this._reactName=n,this._targetInst=s,this.type=i,this.nativeEvent=r,this.target=a,this.currentTarget=null;for(var o in t)t.hasOwnProperty(o)&&(n=t[o],this[o]=n?n(r):r[o]);return this.isDefaultPrevented=(r.defaultPrevented!=null?r.defaultPrevented:r.returnValue===!1)?Sc:$v,this.isPropagationStopped=$v,this}return Mt(e.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Sc)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Sc)},persist:function(){},isPersistent:Sc}),e}var Vr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Tu=On(Vr),Sl=Mt({},Vr,{view:0,detail:0}),eM=On(Sl),ud,fd,Bo,bu=Mt({},Sl,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Hp,button:0,buttons:0,relatedTarget:function(t){return t.relatedTarget===void 0?t.fromElement===t.srcElement?t.toElement:t.fromElement:t.relatedTarget},movementX:function(t){return"movementX"in t?t.movementX:(t!==Bo&&(Bo&&t.type==="mousemove"?(ud=t.screenX-Bo.screenX,fd=t.screenY-Bo.screenY):fd=ud=0,Bo=t),ud)},movementY:function(t){return"movementY"in t?t.movementY:fd}}),e0=On(bu),tM=Mt({},bu,{dataTransfer:0}),nM=On(tM),iM=Mt({},Sl,{relatedTarget:0}),hd=On(iM),sM=Mt({},Vr,{animationName:0,elapsedTime:0,pseudoElement:0}),rM=On(sM),aM=Mt({},Vr,{clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}}),oM=On(aM),lM=Mt({},Vr,{data:0}),t0=On(lM),cM={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},uM={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},fM={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function hM(t){var e=this.nativeEvent;return e.getModifierState?e.getModifierState(t):(t=fM[t])?!!e[t]:!1}function Hp(){return hM}var dM=Mt({},Sl,{key:function(t){if(t.key){var e=cM[t.key]||t.key;if(e!=="Unidentified")return e}return t.type==="keypress"?(t=Oc(t),t===13?"Enter":String.fromCharCode(t)):t.type==="keydown"||t.type==="keyup"?uM[t.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Hp,charCode:function(t){return t.type==="keypress"?Oc(t):0},keyCode:function(t){return t.type==="keydown"||t.type==="keyup"?t.keyCode:0},which:function(t){return t.type==="keypress"?Oc(t):t.type==="keydown"||t.type==="keyup"?t.keyCode:0}}),pM=On(dM),mM=Mt({},bu,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),n0=On(mM),gM=Mt({},Sl,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Hp}),vM=On(gM),xM=Mt({},Vr,{propertyName:0,elapsedTime:0,pseudoElement:0}),yM=On(xM),_M=Mt({},bu,{deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:0,deltaMode:0}),SM=On(_M),AM=Mt({},Vr,{newState:0,oldState:0}),MM=On(AM),EM=[9,13,27,32],Vp=ds&&"CompositionEvent"in window,ko=null;ds&&"documentMode"in document&&(ko=document.documentMode);var TM=ds&&"TextEvent"in window&&!ko,zx=ds&&(!Vp||ko&&8<ko&&11>=ko),i0=" ",s0=!1;function Gx(t,e){switch(t){case"keyup":return EM.indexOf(e.keyCode)!==-1;case"keydown":return e.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Hx(t){return t=t.detail,typeof t=="object"&&"data"in t?t.data:null}var Aa=!1;function bM(t,e){switch(t){case"compositionend":return Hx(e);case"keypress":return e.which!==32?null:(s0=!0,i0);case"textInput":return t=e.data,t===i0&&s0?null:t;default:return null}}function wM(t,e){if(Aa)return t==="compositionend"||!Vp&&Gx(t,e)?(t=Fx(),Nc=Gp=zs=null,Aa=!1,t):null;switch(t){case"paste":return null;case"keypress":if(!(e.ctrlKey||e.altKey||e.metaKey)||e.ctrlKey&&e.altKey){if(e.char&&1<e.char.length)return e.char;if(e.which)return String.fromCharCode(e.which)}return null;case"compositionend":return zx&&e.locale!=="ko"?null:e.data;default:return null}}var CM={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function r0(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e==="input"?!!CM[t.type]:e==="textarea"}function Vx(t,e,n,i){Sa?Ba?Ba.push(i):Ba=[i]:Sa=i,e=vu(e,"onChange"),0<e.length&&(n=new Tu("onChange","change",null,n,i),t.push({event:n,listeners:e}))}var Wo=null,sl=null;function RM(t){O_(t,0)}function wu(t){var e=Go(t);if(Bx(e))return t}function a0(t,e){if(t==="change")return e}var kx=!1;ds&&(ds?(Mc="oninput"in document,Mc||(dd=document.createElement("div"),dd.setAttribute("oninput","return;"),Mc=typeof dd.oninput=="function"),Ac=Mc):Ac=!1,kx=Ac&&(!document.documentMode||9<document.documentMode));var Ac,Mc,dd;function o0(){Wo&&(Wo.detachEvent("onpropertychange",Wx),sl=Wo=null)}function Wx(t){if(t.propertyName==="value"&&wu(sl)){var e=[];Vx(e,sl,t,zp(t)),Ox(RM,e)}}function DM(t,e,n){t==="focusin"?(o0(),Wo=e,sl=n,Wo.attachEvent("onpropertychange",Wx)):t==="focusout"&&o0()}function UM(t){if(t==="selectionchange"||t==="keyup"||t==="keydown")return wu(sl)}function BM(t,e){if(t==="click")return wu(e)}function IM(t,e){if(t==="input"||t==="change")return wu(e)}function PM(t,e){return t===e&&(t!==0||1/t===1/e)||t!==t&&e!==e}var Kn=typeof Object.is=="function"?Object.is:PM;function rl(t,e){if(Kn(t,e))return!0;if(typeof t!="object"||t===null||typeof e!="object"||e===null)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(i=0;i<n.length;i++){var s=n[i];if(!Wd.call(e,s)||!Kn(t[s],e[s]))return!1}return!0}function l0(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function c0(t,e){var n=l0(t);t=0;for(var i;n;){if(n.nodeType===3){if(i=t+n.textContent.length,t<=e&&i>=e)return{node:n,offset:e-t};t=i}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=l0(n)}}function Xx(t,e){return t&&e?t===e?!0:t&&t.nodeType===3?!1:e&&e.nodeType===3?Xx(t,e.parentNode):"contains"in t?t.contains(e):t.compareDocumentPosition?!!(t.compareDocumentPosition(e)&16):!1:!1}function Yx(t){t=t!=null&&t.ownerDocument!=null&&t.ownerDocument.defaultView!=null?t.ownerDocument.defaultView:window;for(var e=$c(t.document);e instanceof t.HTMLIFrameElement;){try{var n=typeof e.contentWindow.location.href=="string"}catch{n=!1}if(n)t=e.contentWindow;else break;e=$c(t.document)}return e}function kp(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&(e==="input"&&(t.type==="text"||t.type==="search"||t.type==="tel"||t.type==="url"||t.type==="password")||e==="textarea"||t.contentEditable==="true")}var LM=ds&&"documentMode"in document&&11>=document.documentMode,Ma=null,Jd=null,Xo=null,jd=!1;function u0(t,e,n){var i=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;jd||Ma==null||Ma!==$c(i)||(i=Ma,"selectionStart"in i&&kp(i)?i={start:i.selectionStart,end:i.selectionEnd}:(i=(i.ownerDocument&&i.ownerDocument.defaultView||window).getSelection(),i={anchorNode:i.anchorNode,anchorOffset:i.anchorOffset,focusNode:i.focusNode,focusOffset:i.focusOffset}),Xo&&rl(Xo,i)||(Xo=i,i=vu(Jd,"onSelect"),0<i.length&&(e=new Tu("onSelect","select",null,e,n),t.push({event:e,listeners:i}),e.target=Ma)))}function br(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n}var Ea={animationend:br("Animation","AnimationEnd"),animationiteration:br("Animation","AnimationIteration"),animationstart:br("Animation","AnimationStart"),transitionrun:br("Transition","TransitionRun"),transitionstart:br("Transition","TransitionStart"),transitioncancel:br("Transition","TransitionCancel"),transitionend:br("Transition","TransitionEnd")},pd={},qx={};ds&&(qx=document.createElement("div").style,"AnimationEvent"in window||(delete Ea.animationend.animation,delete Ea.animationiteration.animation,delete Ea.animationstart.animation),"TransitionEvent"in window||delete Ea.transitionend.transition);function kr(t){if(pd[t])return pd[t];if(!Ea[t])return t;var e=Ea[t],n;for(n in e)if(e.hasOwnProperty(n)&&n in qx)return pd[t]=e[n];return t}var Qx=kr("animationend"),Zx=kr("animationiteration"),Kx=kr("animationstart"),NM=kr("transitionrun"),OM=kr("transitionstart"),FM=kr("transitioncancel"),Jx=kr("transitionend"),jx=new Map,$d="abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");$d.push("scrollEnd");function Ai(t,e){jx.set(t,e),Hr(e,[t])}var eu=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)},ri=[],Ta=0,Wp=0;function Cu(){for(var t=Ta,e=Wp=Ta=0;e<t;){var n=ri[e];ri[e++]=null;var i=ri[e];ri[e++]=null;var s=ri[e];ri[e++]=null;var r=ri[e];if(ri[e++]=null,i!==null&&s!==null){var a=i.pending;a===null?s.next=s:(s.next=a.next,a.next=s),i.pending=s}r!==0&&$x(n,s,r)}}function Ru(t,e,n,i){ri[Ta++]=t,ri[Ta++]=e,ri[Ta++]=n,ri[Ta++]=i,Wp|=i,t.lanes|=i,t=t.alternate,t!==null&&(t.lanes|=i)}function Xp(t,e,n,i){return Ru(t,e,n,i),tu(t)}function Wr(t,e){return Ru(t,null,null,e),tu(t)}function $x(t,e,n){t.lanes|=n;var i=t.alternate;i!==null&&(i.lanes|=n);for(var s=!1,r=t.return;r!==null;)r.childLanes|=n,i=r.alternate,i!==null&&(i.childLanes|=n),r.tag===22&&(t=r.stateNode,t===null||t._visibility&1||(s=!0)),t=r,r=r.return;return t.tag===3?(r=t.stateNode,s&&e!==null&&(s=31-Qn(n),t=r.hiddenUpdates,i=t[s],i===null?t[s]=[e]:i.push(e),e.lane=n|536870912),r):null}function tu(t){if(50<el)throw el=0,_p=null,Error(J(185));for(var e=t.return;e!==null;)t=e,e=t.return;return t.tag===3?t.stateNode:null}var ba={};function zM(t,e,n,i){this.tag=t,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=e,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=i,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Wn(t,e,n,i){return new zM(t,e,n,i)}function Yp(t){return t=t.prototype,!(!t||!t.isReactComponent)}function us(t,e){var n=t.alternate;return n===null?(n=Wn(t.tag,e,t.key,t.mode),n.elementType=t.elementType,n.type=t.type,n.stateNode=t.stateNode,n.alternate=t,t.alternate=n):(n.pendingProps=e,n.type=t.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=t.flags&65011712,n.childLanes=t.childLanes,n.lanes=t.lanes,n.child=t.child,n.memoizedProps=t.memoizedProps,n.memoizedState=t.memoizedState,n.updateQueue=t.updateQueue,e=t.dependencies,n.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext},n.sibling=t.sibling,n.index=t.index,n.ref=t.ref,n.refCleanup=t.refCleanup,n}function ey(t,e){t.flags&=65011714;var n=t.alternate;return n===null?(t.childLanes=0,t.lanes=e,t.child=null,t.subtreeFlags=0,t.memoizedProps=null,t.memoizedState=null,t.updateQueue=null,t.dependencies=null,t.stateNode=null):(t.childLanes=n.childLanes,t.lanes=n.lanes,t.child=n.child,t.subtreeFlags=0,t.deletions=null,t.memoizedProps=n.memoizedProps,t.memoizedState=n.memoizedState,t.updateQueue=n.updateQueue,t.type=n.type,e=n.dependencies,t.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),t}function Fc(t,e,n,i,s,r){var a=0;if(i=t,typeof t=="function")Yp(t)&&(a=1);else if(typeof t=="string")a=VE(t,n,Gi.current)?26:t==="html"||t==="head"||t==="body"?27:5;else e:switch(t){case Gd:return t=Wn(31,n,e,s),t.elementType=Gd,t.lanes=r,t;case xa:return Br(n.children,s,r,e);case yx:a=8,s|=24;break;case Od:return t=Wn(12,n,e,s|2),t.elementType=Od,t.lanes=r,t;case Fd:return t=Wn(13,n,e,s),t.elementType=Fd,t.lanes=r,t;case zd:return t=Wn(19,n,e,s),t.elementType=zd,t.lanes=r,t;default:if(typeof t=="object"&&t!==null)switch(t.$$typeof){case os:a=10;break e;case _x:a=9;break e;case Bp:a=11;break e;case Ip:a=14;break e;case Bs:a=16,i=null;break e}a=29,n=Error(J(130,t===null?"null":typeof t,"")),i=null}return e=Wn(a,n,e,s),e.elementType=t,e.type=i,e.lanes=r,e}function Br(t,e,n,i){return t=Wn(7,t,i,e),t.lanes=n,t}function md(t,e,n){return t=Wn(6,t,null,e),t.lanes=n,t}function ty(t){var e=Wn(18,null,null,0);return e.stateNode=t,e}function gd(t,e,n){return e=Wn(4,t.children!==null?t.children:[],t.key,e),e.lanes=n,e.stateNode={containerInfo:t.containerInfo,pendingChildren:null,implementation:t.implementation},e}var f0=new WeakMap;function ui(t,e){if(typeof t=="object"&&t!==null){var n=f0.get(t);return n!==void 0?n:(e={value:t,source:e,stack:Yv(e)},f0.set(t,e),e)}return{value:t,source:e,stack:Yv(e)}}var wa=[],Ca=0,nu=null,al=0,oi=[],li=0,js=null,Oi=1,Fi="";function rs(t,e){wa[Ca++]=al,wa[Ca++]=nu,nu=t,al=e}function ny(t,e,n){oi[li++]=Oi,oi[li++]=Fi,oi[li++]=js,js=t;var i=Oi;t=Fi;var s=32-Qn(i)-1;i&=~(1<<s),n+=1;var r=32-Qn(e)+s;if(30<r){var a=s-s%5;r=(i&(1<<a)-1).toString(32),i>>=a,s-=a,Oi=1<<32-Qn(e)+s|n<<s|i,Fi=r+t}else Oi=1<<r|n<<s|i,Fi=t}function qp(t){t.return!==null&&(rs(t,1),ny(t,1,0))}function Qp(t){for(;t===nu;)nu=wa[--Ca],wa[Ca]=null,al=wa[--Ca],wa[Ca]=null;for(;t===js;)js=oi[--li],oi[li]=null,Fi=oi[--li],oi[li]=null,Oi=oi[--li],oi[li]=null}function iy(t,e){oi[li++]=Oi,oi[li++]=Fi,oi[li++]=js,Oi=e.id,Fi=e.overflow,js=t}var pn=null,At=null,qe=!1,Ws=null,fi=!1,ep=Error(J(519));function $s(t){var e=Error(J(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?"text":"HTML",""));throw ol(ui(e,t)),ep}function h0(t){var e=t.stateNode,n=t.type,i=t.memoizedProps;switch(e[dn]=t,e[Nn]=i,n){case"dialog":He("cancel",e),He("close",e);break;case"iframe":case"object":case"embed":He("load",e);break;case"video":case"audio":for(n=0;n<fl.length;n++)He(fl[n],e);break;case"source":He("error",e);break;case"img":case"image":case"link":He("error",e),He("load",e);break;case"details":He("toggle",e);break;case"input":He("invalid",e),Ix(e,i.value,i.defaultValue,i.checked,i.defaultChecked,i.type,i.name,!0);break;case"select":He("invalid",e);break;case"textarea":He("invalid",e),Lx(e,i.value,i.defaultValue,i.children)}n=i.children,typeof n!="string"&&typeof n!="number"&&typeof n!="bigint"||e.textContent===""+n||i.suppressHydrationWarning===!0||z_(e.textContent,n)?(i.popover!=null&&(He("beforetoggle",e),He("toggle",e)),i.onScroll!=null&&He("scroll",e),i.onScrollEnd!=null&&He("scrollend",e),i.onClick!=null&&(e.onclick=ls),e=!0):e=!1,e||$s(t,!0)}function d0(t){for(pn=t.return;pn;)switch(pn.tag){case 5:case 31:case 13:fi=!1;return;case 27:case 3:fi=!0;return;default:pn=pn.return}}function pa(t){if(t!==pn)return!1;if(!qe)return d0(t),qe=!0,!1;var e=t.tag,n;if((n=e!==3&&e!==27)&&((n=e===5)&&(n=t.type,n=!(n!=="form"&&n!=="button")||Tp(t.type,t.memoizedProps)),n=!n),n&&At&&$s(t),d0(t),e===13){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));At=tx(t)}else if(e===31){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));At=tx(t)}else e===27?(e=At,ir(t.type)?(t=Rp,Rp=null,At=t):At=e):At=pn?di(t.stateNode.nextSibling):null;return!0}function Nr(){At=pn=null,qe=!1}function vd(){var t=Ws;return t!==null&&(Pn===null?Pn=t:Pn.push.apply(Pn,t),Ws=null),t}function ol(t){Ws===null?Ws=[t]:Ws.push(t)}var tp=Hi(null),Xr=null,cs=null;function Ps(t,e,n){vt(tp,e._currentValue),e._currentValue=n}function fs(t){t._currentValue=tp.current,ln(tp)}function np(t,e,n){for(;t!==null;){var i=t.alternate;if((t.childLanes&e)!==e?(t.childLanes|=e,i!==null&&(i.childLanes|=e)):i!==null&&(i.childLanes&e)!==e&&(i.childLanes|=e),t===n)break;t=t.return}}function ip(t,e,n,i){var s=t.child;for(s!==null&&(s.return=t);s!==null;){var r=s.dependencies;if(r!==null){var a=s.child;r=r.firstContext;e:for(;r!==null;){var o=r;r=s;for(var l=0;l<e.length;l++)if(o.context===e[l]){r.lanes|=n,o=r.alternate,o!==null&&(o.lanes|=n),np(r.return,n,t),i||(a=null);break e}r=o.next}}else if(s.tag===18){if(a=s.return,a===null)throw Error(J(341));a.lanes|=n,r=a.alternate,r!==null&&(r.lanes|=n),np(a,n,t),a=null}else a=s.child;if(a!==null)a.return=s;else for(a=s;a!==null;){if(a===t){a=null;break}if(s=a.sibling,s!==null){s.return=a.return,a=s;break}a=a.return}s=a}}function Ka(t,e,n,i){t=null;for(var s=e,r=!1;s!==null;){if(!r){if((s.flags&524288)!==0)r=!0;else if((s.flags&262144)!==0)break}if(s.tag===10){var a=s.alternate;if(a===null)throw Error(J(387));if(a=a.memoizedProps,a!==null){var o=s.type;Kn(s.pendingProps.value,a.value)||(t!==null?t.push(o):t=[o])}}else if(s===Zc.current){if(a=s.alternate,a===null)throw Error(J(387));a.memoizedState.memoizedState!==s.memoizedState.memoizedState&&(t!==null?t.push(dl):t=[dl])}s=s.return}t!==null&&ip(e,t,n,i),e.flags|=262144}function iu(t){for(t=t.firstContext;t!==null;){if(!Kn(t.context._currentValue,t.memoizedValue))return!0;t=t.next}return!1}function Or(t){Xr=t,cs=null,t=t.dependencies,t!==null&&(t.firstContext=null)}function mn(t){return sy(Xr,t)}function Ec(t,e){return Xr===null&&Or(t),sy(t,e)}function sy(t,e){var n=e._currentValue;if(e={context:e,memoizedValue:n,next:null},cs===null){if(t===null)throw Error(J(308));cs=e,t.dependencies={lanes:0,firstContext:e},t.flags|=524288}else cs=cs.next=e;return n}var GM=typeof AbortController<"u"?AbortController:function(){var t=[],e=this.signal={aborted:!1,addEventListener:function(n,i){t.push(i)}};this.abort=function(){e.aborted=!0,t.forEach(function(n){return n()})}},HM=tn.unstable_scheduleCallback,VM=tn.unstable_NormalPriority,qt={$$typeof:os,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function Zp(){return{controller:new GM,data:new Map,refCount:0}}function Al(t){t.refCount--,t.refCount===0&&HM(VM,function(){t.controller.abort()})}var Yo=null,sp=0,Ga=0,Ia=null;function kM(t,e){if(Yo===null){var n=Yo=[];sp=0,Ga=_m(),Ia={status:"pending",value:void 0,then:function(i){n.push(i)}}}return sp++,e.then(p0,p0),e}function p0(){if(--sp===0&&Yo!==null){Ia!==null&&(Ia.status="fulfilled");var t=Yo;Yo=null,Ga=0,Ia=null;for(var e=0;e<t.length;e++)(0,t[e])()}}function WM(t,e){var n=[],i={status:"pending",value:null,reason:null,then:function(s){n.push(s)}};return t.then(function(){i.status="fulfilled",i.value=e;for(var s=0;s<n.length;s++)(0,n[s])(e)},function(s){for(i.status="rejected",i.reason=s,s=0;s<n.length;s++)(0,n[s])(void 0)}),i}var m0=Be.S;Be.S=function(t,e){x_=Yn(),typeof e=="object"&&e!==null&&typeof e.then=="function"&&kM(t,e),m0!==null&&m0(t,e)};var Ir=Hi(null);function Kp(){var t=Ir.current;return t!==null?t:pt.pooledCache}function zc(t,e){e===null?vt(Ir,Ir.current):vt(Ir,e.pool)}function ry(){var t=Kp();return t===null?null:{parent:qt._currentValue,pool:t}}var Ja=Error(J(460)),Jp=Error(J(474)),Du=Error(J(542)),su={then:function(){}};function g0(t){return t=t.status,t==="fulfilled"||t==="rejected"}function ay(t,e,n){switch(n=t[n],n===void 0?t.push(e):n!==e&&(e.then(ls,ls),e=n),e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,x0(t),t;default:if(typeof e.status=="string")e.then(ls,ls);else{if(t=pt,t!==null&&100<t.shellSuspendCounter)throw Error(J(482));t=e,t.status="pending",t.then(function(i){if(e.status==="pending"){var s=e;s.status="fulfilled",s.value=i}},function(i){if(e.status==="pending"){var s=e;s.status="rejected",s.reason=i}})}switch(e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,x0(t),t}throw Pr=e,Ja}}function Rr(t){try{var e=t._init;return e(t._payload)}catch(n){throw n!==null&&typeof n=="object"&&typeof n.then=="function"?(Pr=n,Ja):n}}var Pr=null;function v0(){if(Pr===null)throw Error(J(459));var t=Pr;return Pr=null,t}function x0(t){if(t===Ja||t===Du)throw Error(J(483))}var Pa=null,ll=0;function Tc(t){var e=ll;return ll+=1,Pa===null&&(Pa=[]),ay(Pa,t,e)}function Io(t,e){e=e.props.ref,t.ref=e!==void 0?e:null}function bc(t,e){throw e.$$typeof===D1?Error(J(525)):(t=Object.prototype.toString.call(e),Error(J(31,t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)))}function oy(t){function e(f,g){if(t){var S=f.deletions;S===null?(f.deletions=[g],f.flags|=16):S.push(g)}}function n(f,g){if(!t)return null;for(;g!==null;)e(f,g),g=g.sibling;return null}function i(f){for(var g=new Map;f!==null;)f.key!==null?g.set(f.key,f):g.set(f.index,f),f=f.sibling;return g}function s(f,g){return f=us(f,g),f.index=0,f.sibling=null,f}function r(f,g,S){return f.index=S,t?(S=f.alternate,S!==null?(S=S.index,S<g?(f.flags|=67108866,g):S):(f.flags|=67108866,g)):(f.flags|=1048576,g)}function a(f){return t&&f.alternate===null&&(f.flags|=67108866),f}function o(f,g,S,_){return g===null||g.tag!==6?(g=md(S,f.mode,_),g.return=f,g):(g=s(g,S),g.return=f,g)}function l(f,g,S,_){var T=S.type;return T===xa?h(f,g,S.props.children,_,S.key):g!==null&&(g.elementType===T||typeof T=="object"&&T!==null&&T.$$typeof===Bs&&Rr(T)===g.type)?(g=s(g,S.props),Io(g,S),g.return=f,g):(g=Fc(S.type,S.key,S.props,null,f.mode,_),Io(g,S),g.return=f,g)}function c(f,g,S,_){return g===null||g.tag!==4||g.stateNode.containerInfo!==S.containerInfo||g.stateNode.implementation!==S.implementation?(g=gd(S,f.mode,_),g.return=f,g):(g=s(g,S.children||[]),g.return=f,g)}function h(f,g,S,_,T){return g===null||g.tag!==7?(g=Br(S,f.mode,_,T),g.return=f,g):(g=s(g,S),g.return=f,g)}function p(f,g,S){if(typeof g=="string"&&g!==""||typeof g=="number"||typeof g=="bigint")return g=md(""+g,f.mode,S),g.return=f,g;if(typeof g=="object"&&g!==null){switch(g.$$typeof){case gc:return S=Fc(g.type,g.key,g.props,null,f.mode,S),Io(S,g),S.return=f,S;case Fo:return g=gd(g,f.mode,S),g.return=f,g;case Bs:return g=Rr(g),p(f,g,S)}if(zo(g)||Uo(g))return g=Br(g,f.mode,S,null),g.return=f,g;if(typeof g.then=="function")return p(f,Tc(g),S);if(g.$$typeof===os)return p(f,Ec(f,g),S);bc(f,g)}return null}function u(f,g,S,_){var T=g!==null?g.key:null;if(typeof S=="string"&&S!==""||typeof S=="number"||typeof S=="bigint")return T!==null?null:o(f,g,""+S,_);if(typeof S=="object"&&S!==null){switch(S.$$typeof){case gc:return S.key===T?l(f,g,S,_):null;case Fo:return S.key===T?c(f,g,S,_):null;case Bs:return S=Rr(S),u(f,g,S,_)}if(zo(S)||Uo(S))return T!==null?null:h(f,g,S,_,null);if(typeof S.then=="function")return u(f,g,Tc(S),_);if(S.$$typeof===os)return u(f,g,Ec(f,S),_);bc(f,S)}return null}function d(f,g,S,_,T){if(typeof _=="string"&&_!==""||typeof _=="number"||typeof _=="bigint")return f=f.get(S)||null,o(g,f,""+_,T);if(typeof _=="object"&&_!==null){switch(_.$$typeof){case gc:return f=f.get(_.key===null?S:_.key)||null,l(g,f,_,T);case Fo:return f=f.get(_.key===null?S:_.key)||null,c(g,f,_,T);case Bs:return _=Rr(_),d(f,g,S,_,T)}if(zo(_)||Uo(_))return f=f.get(S)||null,h(g,f,_,T,null);if(typeof _.then=="function")return d(f,g,S,Tc(_),T);if(_.$$typeof===os)return d(f,g,S,Ec(g,_),T);bc(g,_)}return null}function v(f,g,S,_){for(var T=null,b=null,w=g,x=g=0,E=null;w!==null&&x<S.length;x++){w.index>x?(E=w,w=null):E=w.sibling;var R=u(f,w,S[x],_);if(R===null){w===null&&(w=E);break}t&&w&&R.alternate===null&&e(f,w),g=r(R,g,x),b===null?T=R:b.sibling=R,b=R,w=E}if(x===S.length)return n(f,w),qe&&rs(f,x),T;if(w===null){for(;x<S.length;x++)w=p(f,S[x],_),w!==null&&(g=r(w,g,x),b===null?T=w:b.sibling=w,b=w);return qe&&rs(f,x),T}for(w=i(w);x<S.length;x++)E=d(w,f,x,S[x],_),E!==null&&(t&&E.alternate!==null&&w.delete(E.key===null?x:E.key),g=r(E,g,x),b===null?T=E:b.sibling=E,b=E);return t&&w.forEach(function(D){return e(f,D)}),qe&&rs(f,x),T}function M(f,g,S,_){if(S==null)throw Error(J(151));for(var T=null,b=null,w=g,x=g=0,E=null,R=S.next();w!==null&&!R.done;x++,R=S.next()){w.index>x?(E=w,w=null):E=w.sibling;var D=u(f,w,R.value,_);if(D===null){w===null&&(w=E);break}t&&w&&D.alternate===null&&e(f,w),g=r(D,g,x),b===null?T=D:b.sibling=D,b=D,w=E}if(R.done)return n(f,w),qe&&rs(f,x),T;if(w===null){for(;!R.done;x++,R=S.next())R=p(f,R.value,_),R!==null&&(g=r(R,g,x),b===null?T=R:b.sibling=R,b=R);return qe&&rs(f,x),T}for(w=i(w);!R.done;x++,R=S.next())R=d(w,f,x,R.value,_),R!==null&&(t&&R.alternate!==null&&w.delete(R.key===null?x:R.key),g=r(R,g,x),b===null?T=R:b.sibling=R,b=R);return t&&w.forEach(function(L){return e(f,L)}),qe&&rs(f,x),T}function m(f,g,S,_){if(typeof S=="object"&&S!==null&&S.type===xa&&S.key===null&&(S=S.props.children),typeof S=="object"&&S!==null){switch(S.$$typeof){case gc:e:{for(var T=S.key;g!==null;){if(g.key===T){if(T=S.type,T===xa){if(g.tag===7){n(f,g.sibling),_=s(g,S.props.children),_.return=f,f=_;break e}}else if(g.elementType===T||typeof T=="object"&&T!==null&&T.$$typeof===Bs&&Rr(T)===g.type){n(f,g.sibling),_=s(g,S.props),Io(_,S),_.return=f,f=_;break e}n(f,g);break}else e(f,g);g=g.sibling}S.type===xa?(_=Br(S.props.children,f.mode,_,S.key),_.return=f,f=_):(_=Fc(S.type,S.key,S.props,null,f.mode,_),Io(_,S),_.return=f,f=_)}return a(f);case Fo:e:{for(T=S.key;g!==null;){if(g.key===T)if(g.tag===4&&g.stateNode.containerInfo===S.containerInfo&&g.stateNode.implementation===S.implementation){n(f,g.sibling),_=s(g,S.children||[]),_.return=f,f=_;break e}else{n(f,g);break}else e(f,g);g=g.sibling}_=gd(S,f.mode,_),_.return=f,f=_}return a(f);case Bs:return S=Rr(S),m(f,g,S,_)}if(zo(S))return v(f,g,S,_);if(Uo(S)){if(T=Uo(S),typeof T!="function")throw Error(J(150));return S=T.call(S),M(f,g,S,_)}if(typeof S.then=="function")return m(f,g,Tc(S),_);if(S.$$typeof===os)return m(f,g,Ec(f,S),_);bc(f,S)}return typeof S=="string"&&S!==""||typeof S=="number"||typeof S=="bigint"?(S=""+S,g!==null&&g.tag===6?(n(f,g.sibling),_=s(g,S),_.return=f,f=_):(n(f,g),_=md(S,f.mode,_),_.return=f,f=_),a(f)):n(f,g)}return function(f,g,S,_){try{ll=0;var T=m(f,g,S,_);return Pa=null,T}catch(w){if(w===Ja||w===Du)throw w;var b=Wn(29,w,null,f.mode);return b.lanes=_,b.return=f,b}}}var Fr=oy(!0),ly=oy(!1),Is=!1;function jp(t){t.updateQueue={baseState:t.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function rp(t,e){t=t.updateQueue,e.updateQueue===t&&(e.updateQueue={baseState:t.baseState,firstBaseUpdate:t.firstBaseUpdate,lastBaseUpdate:t.lastBaseUpdate,shared:t.shared,callbacks:null})}function Xs(t){return{lane:t,tag:0,payload:null,callback:null,next:null}}function Ys(t,e,n){var i=t.updateQueue;if(i===null)return null;if(i=i.shared,(nt&2)!==0){var s=i.pending;return s===null?e.next=e:(e.next=s.next,s.next=e),i.pending=e,e=tu(t),$x(t,null,n),e}return Ru(t,i,e,n),tu(t)}function qo(t,e,n){if(e=e.updateQueue,e!==null&&(e=e.shared,(n&4194048)!==0)){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,bx(t,n)}}function xd(t,e){var n=t.updateQueue,i=t.alternate;if(i!==null&&(i=i.updateQueue,n===i)){var s=null,r=null;if(n=n.firstBaseUpdate,n!==null){do{var a={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};r===null?s=r=a:r=r.next=a,n=n.next}while(n!==null);r===null?s=r=e:r=r.next=e}else s=r=e;n={baseState:i.baseState,firstBaseUpdate:s,lastBaseUpdate:r,shared:i.shared,callbacks:i.callbacks},t.updateQueue=n;return}t=n.lastBaseUpdate,t===null?n.firstBaseUpdate=e:t.next=e,n.lastBaseUpdate=e}var ap=!1;function Qo(){if(ap){var t=Ia;if(t!==null)throw t}}function Zo(t,e,n,i){ap=!1;var s=t.updateQueue;Is=!1;var r=s.firstBaseUpdate,a=s.lastBaseUpdate,o=s.shared.pending;if(o!==null){s.shared.pending=null;var l=o,c=l.next;l.next=null,a===null?r=c:a.next=c,a=l;var h=t.alternate;h!==null&&(h=h.updateQueue,o=h.lastBaseUpdate,o!==a&&(o===null?h.firstBaseUpdate=c:o.next=c,h.lastBaseUpdate=l))}if(r!==null){var p=s.baseState;a=0,h=c=l=null,o=r;do{var u=o.lane&-536870913,d=u!==o.lane;if(d?(Xe&u)===u:(i&u)===u){u!==0&&u===Ga&&(ap=!0),h!==null&&(h=h.next={lane:0,tag:o.tag,payload:o.payload,callback:null,next:null});e:{var v=t,M=o;u=e;var m=n;switch(M.tag){case 1:if(v=M.payload,typeof v=="function"){p=v.call(m,p,u);break e}p=v;break e;case 3:v.flags=v.flags&-65537|128;case 0:if(v=M.payload,u=typeof v=="function"?v.call(m,p,u):v,u==null)break e;p=Mt({},p,u);break e;case 2:Is=!0}}u=o.callback,u!==null&&(t.flags|=64,d&&(t.flags|=8192),d=s.callbacks,d===null?s.callbacks=[u]:d.push(u))}else d={lane:u,tag:o.tag,payload:o.payload,callback:o.callback,next:null},h===null?(c=h=d,l=p):h=h.next=d,a|=u;if(o=o.next,o===null){if(o=s.shared.pending,o===null)break;d=o,o=d.next,d.next=null,s.lastBaseUpdate=d,s.shared.pending=null}}while(!0);h===null&&(l=p),s.baseState=l,s.firstBaseUpdate=c,s.lastBaseUpdate=h,r===null&&(s.shared.lanes=0),tr|=a,t.lanes=a,t.memoizedState=p}}function cy(t,e){if(typeof t!="function")throw Error(J(191,t));t.call(e)}function uy(t,e){var n=t.callbacks;if(n!==null)for(t.callbacks=null,t=0;t<n.length;t++)cy(n[t],e)}var Ha=Hi(null),ru=Hi(0);function y0(t,e){t=vs,vt(ru,t),vt(Ha,e),vs=t|e.baseLanes}function op(){vt(ru,vs),vt(Ha,Ha.current)}function $p(){vs=ru.current,ln(Ha),ln(ru)}var Jn=Hi(null),hi=null;function Ls(t){var e=t.alternate;vt(zt,zt.current&1),vt(Jn,t),hi===null&&(e===null||Ha.current!==null||e.memoizedState!==null)&&(hi=t)}function lp(t){vt(zt,zt.current),vt(Jn,t),hi===null&&(hi=t)}function fy(t){t.tag===22?(vt(zt,zt.current),vt(Jn,t),hi===null&&(hi=t)):Ns(t)}function Ns(){vt(zt,zt.current),vt(Jn,Jn.current)}function kn(t){ln(Jn),hi===t&&(hi=null),ln(zt)}var zt=Hi(0);function au(t){for(var e=t;e!==null;){if(e.tag===13){var n=e.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||wp(n)||Cp(n)))return e}else if(e.tag===19&&(e.memoizedProps.revealOrder==="forwards"||e.memoizedProps.revealOrder==="backwards"||e.memoizedProps.revealOrder==="unstable_legacy-backwards"||e.memoizedProps.revealOrder==="together")){if((e.flags&128)!==0)return e}else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return null;e=e.return}e.sibling.return=e.return,e=e.sibling}return null}var ps=0,Fe=null,ut=null,Xt=null,ou=!1,La=!1,zr=!1,lu=0,cl=0,Na=null,XM=0;function Pt(){throw Error(J(321))}function em(t,e){if(e===null)return!1;for(var n=0;n<e.length&&n<t.length;n++)if(!Kn(t[n],e[n]))return!1;return!0}function tm(t,e,n,i,s,r){return ps=r,Fe=e,e.memoizedState=null,e.updateQueue=null,e.lanes=0,Be.H=t===null||t.memoizedState===null?Vy:hm,zr=!1,r=n(i,s),zr=!1,La&&(r=dy(e,n,i,s)),hy(t),r}function hy(t){Be.H=ul;var e=ut!==null&&ut.next!==null;if(ps=0,Xt=ut=Fe=null,ou=!1,cl=0,Na=null,e)throw Error(J(300));t===null||Qt||(t=t.dependencies,t!==null&&iu(t)&&(Qt=!0))}function dy(t,e,n,i){Fe=t;var s=0;do{if(La&&(Na=null),cl=0,La=!1,25<=s)throw Error(J(301));if(s+=1,Xt=ut=null,t.updateQueue!=null){var r=t.updateQueue;r.lastEffect=null,r.events=null,r.stores=null,r.memoCache!=null&&(r.memoCache.index=0)}Be.H=ky,r=e(n,i)}while(La);return r}function YM(){var t=Be.H,e=t.useState()[0];return e=typeof e.then=="function"?Ml(e):e,t=t.useState()[0],(ut!==null?ut.memoizedState:null)!==t&&(Fe.flags|=1024),e}function nm(){var t=lu!==0;return lu=0,t}function im(t,e,n){e.updateQueue=t.updateQueue,e.flags&=-2053,t.lanes&=~n}function sm(t){if(ou){for(t=t.memoizedState;t!==null;){var e=t.queue;e!==null&&(e.pending=null),t=t.next}ou=!1}ps=0,Xt=ut=Fe=null,La=!1,cl=lu=0,Na=null}function Cn(){var t={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Xt===null?Fe.memoizedState=Xt=t:Xt=Xt.next=t,Xt}function Gt(){if(ut===null){var t=Fe.alternate;t=t!==null?t.memoizedState:null}else t=ut.next;var e=Xt===null?Fe.memoizedState:Xt.next;if(e!==null)Xt=e,ut=t;else{if(t===null)throw Fe.alternate===null?Error(J(467)):Error(J(310));ut=t,t={memoizedState:ut.memoizedState,baseState:ut.baseState,baseQueue:ut.baseQueue,queue:ut.queue,next:null},Xt===null?Fe.memoizedState=Xt=t:Xt=Xt.next=t}return Xt}function Uu(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function Ml(t){var e=cl;return cl+=1,Na===null&&(Na=[]),t=ay(Na,t,e),e=Fe,(Xt===null?e.memoizedState:Xt.next)===null&&(e=e.alternate,Be.H=e===null||e.memoizedState===null?Vy:hm),t}function Bu(t){if(t!==null&&typeof t=="object"){if(typeof t.then=="function")return Ml(t);if(t.$$typeof===os)return mn(t)}throw Error(J(438,String(t)))}function rm(t){var e=null,n=Fe.updateQueue;if(n!==null&&(e=n.memoCache),e==null){var i=Fe.alternate;i!==null&&(i=i.updateQueue,i!==null&&(i=i.memoCache,i!=null&&(e={data:i.data.map(function(s){return s.slice()}),index:0})))}if(e==null&&(e={data:[],index:0}),n===null&&(n=Uu(),Fe.updateQueue=n),n.memoCache=e,n=e.data[e.index],n===void 0)for(n=e.data[e.index]=Array(t),i=0;i<t;i++)n[i]=U1;return e.index++,n}function ms(t,e){return typeof e=="function"?e(t):e}function Gc(t){var e=Gt();return am(e,ut,t)}function am(t,e,n){var i=t.queue;if(i===null)throw Error(J(311));i.lastRenderedReducer=n;var s=t.baseQueue,r=i.pending;if(r!==null){if(s!==null){var a=s.next;s.next=r.next,r.next=a}e.baseQueue=s=r,i.pending=null}if(r=t.baseState,s===null)t.memoizedState=r;else{e=s.next;var o=a=null,l=null,c=e,h=!1;do{var p=c.lane&-536870913;if(p!==c.lane?(Xe&p)===p:(ps&p)===p){var u=c.revertLane;if(u===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null}),p===Ga&&(h=!0);else if((ps&u)===u){c=c.next,u===Ga&&(h=!0);continue}else p={lane:0,revertLane:c.revertLane,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=p,a=r):l=l.next=p,Fe.lanes|=u,tr|=u;p=c.action,zr&&n(r,p),r=c.hasEagerState?c.eagerState:n(r,p)}else u={lane:p,revertLane:c.revertLane,gesture:c.gesture,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=u,a=r):l=l.next=u,Fe.lanes|=p,tr|=p;c=c.next}while(c!==null&&c!==e);if(l===null?a=r:l.next=o,!Kn(r,t.memoizedState)&&(Qt=!0,h&&(n=Ia,n!==null)))throw n;t.memoizedState=r,t.baseState=a,t.baseQueue=l,i.lastRenderedState=r}return s===null&&(i.lanes=0),[t.memoizedState,i.dispatch]}function yd(t){var e=Gt(),n=e.queue;if(n===null)throw Error(J(311));n.lastRenderedReducer=t;var i=n.dispatch,s=n.pending,r=e.memoizedState;if(s!==null){n.pending=null;var a=s=s.next;do r=t(r,a.action),a=a.next;while(a!==s);Kn(r,e.memoizedState)||(Qt=!0),e.memoizedState=r,e.baseQueue===null&&(e.baseState=r),n.lastRenderedState=r}return[r,i]}function py(t,e,n){var i=Fe,s=Gt(),r=qe;if(r){if(n===void 0)throw Error(J(407));n=n()}else n=e();var a=!Kn((ut||s).memoizedState,n);if(a&&(s.memoizedState=n,Qt=!0),s=s.queue,om(vy.bind(null,i,s,t),[t]),s.getSnapshot!==e||a||Xt!==null&&Xt.memoizedState.tag&1){if(i.flags|=2048,Va(9,{destroy:void 0},gy.bind(null,i,s,n,e),null),pt===null)throw Error(J(349));r||(ps&127)!==0||my(i,e,n)}return n}function my(t,e,n){t.flags|=16384,t={getSnapshot:e,value:n},e=Fe.updateQueue,e===null?(e=Uu(),Fe.updateQueue=e,e.stores=[t]):(n=e.stores,n===null?e.stores=[t]:n.push(t))}function gy(t,e,n,i){e.value=n,e.getSnapshot=i,xy(e)&&yy(t)}function vy(t,e,n){return n(function(){xy(e)&&yy(t)})}function xy(t){var e=t.getSnapshot;t=t.value;try{var n=e();return!Kn(t,n)}catch{return!0}}function yy(t){var e=Wr(t,2);e!==null&&Ln(e,t,2)}function cp(t){var e=Cn();if(typeof t=="function"){var n=t;if(t=n(),zr){Fs(!0);try{n()}finally{Fs(!1)}}}return e.memoizedState=e.baseState=t,e.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:t},e}function _y(t,e,n,i){return t.baseState=n,am(t,ut,typeof i=="function"?i:ms)}function qM(t,e,n,i,s){if(Pu(t))throw Error(J(485));if(t=e.action,t!==null){var r={payload:s,action:t,next:null,isTransition:!0,status:"pending",value:null,reason:null,listeners:[],then:function(a){r.listeners.push(a)}};Be.T!==null?n(!0):r.isTransition=!1,i(r),n=e.pending,n===null?(r.next=e.pending=r,Sy(e,r)):(r.next=n.next,e.pending=n.next=r)}}function Sy(t,e){var n=e.action,i=e.payload,s=t.state;if(e.isTransition){var r=Be.T,a={};Be.T=a;try{var o=n(s,i),l=Be.S;l!==null&&l(a,o),_0(t,e,o)}catch(c){up(t,e,c)}finally{r!==null&&a.types!==null&&(r.types=a.types),Be.T=r}}else try{r=n(s,i),_0(t,e,r)}catch(c){up(t,e,c)}}function _0(t,e,n){n!==null&&typeof n=="object"&&typeof n.then=="function"?n.then(function(i){S0(t,e,i)},function(i){return up(t,e,i)}):S0(t,e,n)}function S0(t,e,n){e.status="fulfilled",e.value=n,Ay(e),t.state=n,e=t.pending,e!==null&&(n=e.next,n===e?t.pending=null:(n=n.next,e.next=n,Sy(t,n)))}function up(t,e,n){var i=t.pending;if(t.pending=null,i!==null){i=i.next;do e.status="rejected",e.reason=n,Ay(e),e=e.next;while(e!==i)}t.action=null}function Ay(t){t=t.listeners;for(var e=0;e<t.length;e++)(0,t[e])()}function My(t,e){return e}function A0(t,e){if(qe){var n=pt.formState;if(n!==null){e:{var i=Fe;if(qe){if(At){t:{for(var s=At,r=fi;s.nodeType!==8;){if(!r){s=null;break t}if(s=di(s.nextSibling),s===null){s=null;break t}}r=s.data,s=r==="F!"||r==="F"?s:null}if(s){At=di(s.nextSibling),i=s.data==="F!";break e}}$s(i)}i=!1}i&&(e=n[0])}}return n=Cn(),n.memoizedState=n.baseState=e,i={pending:null,lanes:0,dispatch:null,lastRenderedReducer:My,lastRenderedState:e},n.queue=i,n=zy.bind(null,Fe,i),i.dispatch=n,i=cp(!1),r=fm.bind(null,Fe,!1,i.queue),i=Cn(),s={state:e,dispatch:null,action:t,pending:null},i.queue=s,n=qM.bind(null,Fe,s,r,n),s.dispatch=n,i.memoizedState=t,[e,n,!1]}function M0(t){var e=Gt();return Ey(e,ut,t)}function Ey(t,e,n){if(e=am(t,e,My)[0],t=Gc(ms)[0],typeof e=="object"&&e!==null&&typeof e.then=="function")try{var i=Ml(e)}catch(a){throw a===Ja?Du:a}else i=e;e=Gt();var s=e.queue,r=s.dispatch;return n!==e.memoizedState&&(Fe.flags|=2048,Va(9,{destroy:void 0},QM.bind(null,s,n),null)),[i,r,t]}function QM(t,e){t.action=e}function E0(t){var e=Gt(),n=ut;if(n!==null)return Ey(e,n,t);Gt(),e=e.memoizedState,n=Gt();var i=n.queue.dispatch;return n.memoizedState=t,[e,i,!1]}function Va(t,e,n,i){return t={tag:t,create:n,deps:i,inst:e,next:null},e=Fe.updateQueue,e===null&&(e=Uu(),Fe.updateQueue=e),n=e.lastEffect,n===null?e.lastEffect=t.next=t:(i=n.next,n.next=t,t.next=i,e.lastEffect=t),t}function Ty(){return Gt().memoizedState}function Hc(t,e,n,i){var s=Cn();Fe.flags|=t,s.memoizedState=Va(1|e,{destroy:void 0},n,i===void 0?null:i)}function Iu(t,e,n,i){var s=Gt();i=i===void 0?null:i;var r=s.memoizedState.inst;ut!==null&&i!==null&&em(i,ut.memoizedState.deps)?s.memoizedState=Va(e,r,n,i):(Fe.flags|=t,s.memoizedState=Va(1|e,r,n,i))}function T0(t,e){Hc(8390656,8,t,e)}function om(t,e){Iu(2048,8,t,e)}function ZM(t){Fe.flags|=4;var e=Fe.updateQueue;if(e===null)e=Uu(),Fe.updateQueue=e,e.events=[t];else{var n=e.events;n===null?e.events=[t]:n.push(t)}}function by(t){var e=Gt().memoizedState;return ZM({ref:e,nextImpl:t}),function(){if((nt&2)!==0)throw Error(J(440));return e.impl.apply(void 0,arguments)}}function wy(t,e){return Iu(4,2,t,e)}function Cy(t,e){return Iu(4,4,t,e)}function Ry(t,e){if(typeof e=="function"){t=t();var n=e(t);return function(){typeof n=="function"?n():e(null)}}if(e!=null)return t=t(),e.current=t,function(){e.current=null}}function Dy(t,e,n){n=n!=null?n.concat([t]):null,Iu(4,4,Ry.bind(null,e,t),n)}function lm(){}function Uy(t,e){var n=Gt();e=e===void 0?null:e;var i=n.memoizedState;return e!==null&&em(e,i[1])?i[0]:(n.memoizedState=[t,e],t)}function By(t,e){var n=Gt();e=e===void 0?null:e;var i=n.memoizedState;if(e!==null&&em(e,i[1]))return i[0];if(i=t(),zr){Fs(!0);try{t()}finally{Fs(!1)}}return n.memoizedState=[i,e],i}function cm(t,e,n){return n===void 0||(ps&1073741824)!==0&&(Xe&261930)===0?t.memoizedState=e:(t.memoizedState=n,t=__(),Fe.lanes|=t,tr|=t,n)}function Iy(t,e,n,i){return Kn(n,e)?n:Ha.current!==null?(t=cm(t,n,i),Kn(t,e)||(Qt=!0),t):(ps&42)===0||(ps&1073741824)!==0&&(Xe&261930)===0?(Qt=!0,t.memoizedState=n):(t=__(),Fe.lanes|=t,tr|=t,e)}function Py(t,e,n,i,s){var r=it.p;it.p=r!==0&&8>r?r:8;var a=Be.T,o={};Be.T=o,fm(t,!1,e,n);try{var l=s(),c=Be.S;if(c!==null&&c(o,l),l!==null&&typeof l=="object"&&typeof l.then=="function"){var h=WM(l,i);Ko(t,e,h,Zn(t))}else Ko(t,e,i,Zn(t))}catch(p){Ko(t,e,{then:function(){},status:"rejected",reason:p},Zn())}finally{it.p=r,a!==null&&o.types!==null&&(a.types=o.types),Be.T=a}}function KM(){}function fp(t,e,n,i){if(t.tag!==5)throw Error(J(476));var s=Ly(t).queue;Py(t,s,e,Ur,n===null?KM:function(){return Ny(t),n(i)})}function Ly(t){var e=t.memoizedState;if(e!==null)return e;e={memoizedState:Ur,baseState:Ur,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:Ur},next:null};var n={};return e.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:n},next:null},t.memoizedState=e,t=t.alternate,t!==null&&(t.memoizedState=e),e}function Ny(t){var e=Ly(t);e.next===null&&(e=t.alternate.memoizedState),Ko(t,e.next.queue,{},Zn())}function um(){return mn(dl)}function Oy(){return Gt().memoizedState}function Fy(){return Gt().memoizedState}function JM(t){for(var e=t.return;e!==null;){switch(e.tag){case 24:case 3:var n=Zn();t=Xs(n);var i=Ys(e,t,n);i!==null&&(Ln(i,e,n),qo(i,e,n)),e={cache:Zp()},t.payload=e;return}e=e.return}}function jM(t,e,n){var i=Zn();n={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},Pu(t)?Gy(e,n):(n=Xp(t,e,n,i),n!==null&&(Ln(n,t,i),Hy(n,e,i)))}function zy(t,e,n){var i=Zn();Ko(t,e,n,i)}function Ko(t,e,n,i){var s={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(Pu(t))Gy(e,s);else{var r=t.alternate;if(t.lanes===0&&(r===null||r.lanes===0)&&(r=e.lastRenderedReducer,r!==null))try{var a=e.lastRenderedState,o=r(a,n);if(s.hasEagerState=!0,s.eagerState=o,Kn(o,a))return Ru(t,e,s,0),pt===null&&Cu(),!1}catch{}if(n=Xp(t,e,s,i),n!==null)return Ln(n,t,i),Hy(n,e,i),!0}return!1}function fm(t,e,n,i){if(i={lane:2,revertLane:_m(),gesture:null,action:i,hasEagerState:!1,eagerState:null,next:null},Pu(t)){if(e)throw Error(J(479))}else e=Xp(t,n,i,2),e!==null&&Ln(e,t,2)}function Pu(t){var e=t.alternate;return t===Fe||e!==null&&e===Fe}function Gy(t,e){La=ou=!0;var n=t.pending;n===null?e.next=e:(e.next=n.next,n.next=e),t.pending=e}function Hy(t,e,n){if((n&4194048)!==0){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,bx(t,n)}}var ul={readContext:mn,use:Bu,useCallback:Pt,useContext:Pt,useEffect:Pt,useImperativeHandle:Pt,useLayoutEffect:Pt,useInsertionEffect:Pt,useMemo:Pt,useReducer:Pt,useRef:Pt,useState:Pt,useDebugValue:Pt,useDeferredValue:Pt,useTransition:Pt,useSyncExternalStore:Pt,useId:Pt,useHostTransitionStatus:Pt,useFormState:Pt,useActionState:Pt,useOptimistic:Pt,useMemoCache:Pt,useCacheRefresh:Pt};ul.useEffectEvent=Pt;var Vy={readContext:mn,use:Bu,useCallback:function(t,e){return Cn().memoizedState=[t,e===void 0?null:e],t},useContext:mn,useEffect:T0,useImperativeHandle:function(t,e,n){n=n!=null?n.concat([t]):null,Hc(4194308,4,Ry.bind(null,e,t),n)},useLayoutEffect:function(t,e){return Hc(4194308,4,t,e)},useInsertionEffect:function(t,e){Hc(4,2,t,e)},useMemo:function(t,e){var n=Cn();e=e===void 0?null:e;var i=t();if(zr){Fs(!0);try{t()}finally{Fs(!1)}}return n.memoizedState=[i,e],i},useReducer:function(t,e,n){var i=Cn();if(n!==void 0){var s=n(e);if(zr){Fs(!0);try{n(e)}finally{Fs(!1)}}}else s=e;return i.memoizedState=i.baseState=s,t={pending:null,lanes:0,dispatch:null,lastRenderedReducer:t,lastRenderedState:s},i.queue=t,t=t.dispatch=jM.bind(null,Fe,t),[i.memoizedState,t]},useRef:function(t){var e=Cn();return t={current:t},e.memoizedState=t},useState:function(t){t=cp(t);var e=t.queue,n=zy.bind(null,Fe,e);return e.dispatch=n,[t.memoizedState,n]},useDebugValue:lm,useDeferredValue:function(t,e){var n=Cn();return cm(n,t,e)},useTransition:function(){var t=cp(!1);return t=Py.bind(null,Fe,t.queue,!0,!1),Cn().memoizedState=t,[!1,t]},useSyncExternalStore:function(t,e,n){var i=Fe,s=Cn();if(qe){if(n===void 0)throw Error(J(407));n=n()}else{if(n=e(),pt===null)throw Error(J(349));(Xe&127)!==0||my(i,e,n)}s.memoizedState=n;var r={value:n,getSnapshot:e};return s.queue=r,T0(vy.bind(null,i,r,t),[t]),i.flags|=2048,Va(9,{destroy:void 0},gy.bind(null,i,r,n,e),null),n},useId:function(){var t=Cn(),e=pt.identifierPrefix;if(qe){var n=Fi,i=Oi;n=(i&~(1<<32-Qn(i)-1)).toString(32)+n,e="_"+e+"R_"+n,n=lu++,0<n&&(e+="H"+n.toString(32)),e+="_"}else n=XM++,e="_"+e+"r_"+n.toString(32)+"_";return t.memoizedState=e},useHostTransitionStatus:um,useFormState:A0,useActionState:A0,useOptimistic:function(t){var e=Cn();e.memoizedState=e.baseState=t;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return e.queue=n,e=fm.bind(null,Fe,!0,n),n.dispatch=e,[t,e]},useMemoCache:rm,useCacheRefresh:function(){return Cn().memoizedState=JM.bind(null,Fe)},useEffectEvent:function(t){var e=Cn(),n={impl:t};return e.memoizedState=n,function(){if((nt&2)!==0)throw Error(J(440));return n.impl.apply(void 0,arguments)}}},hm={readContext:mn,use:Bu,useCallback:Uy,useContext:mn,useEffect:om,useImperativeHandle:Dy,useInsertionEffect:wy,useLayoutEffect:Cy,useMemo:By,useReducer:Gc,useRef:Ty,useState:function(){return Gc(ms)},useDebugValue:lm,useDeferredValue:function(t,e){var n=Gt();return Iy(n,ut.memoizedState,t,e)},useTransition:function(){var t=Gc(ms)[0],e=Gt().memoizedState;return[typeof t=="boolean"?t:Ml(t),e]},useSyncExternalStore:py,useId:Oy,useHostTransitionStatus:um,useFormState:M0,useActionState:M0,useOptimistic:function(t,e){var n=Gt();return _y(n,ut,t,e)},useMemoCache:rm,useCacheRefresh:Fy};hm.useEffectEvent=by;var ky={readContext:mn,use:Bu,useCallback:Uy,useContext:mn,useEffect:om,useImperativeHandle:Dy,useInsertionEffect:wy,useLayoutEffect:Cy,useMemo:By,useReducer:yd,useRef:Ty,useState:function(){return yd(ms)},useDebugValue:lm,useDeferredValue:function(t,e){var n=Gt();return ut===null?cm(n,t,e):Iy(n,ut.memoizedState,t,e)},useTransition:function(){var t=yd(ms)[0],e=Gt().memoizedState;return[typeof t=="boolean"?t:Ml(t),e]},useSyncExternalStore:py,useId:Oy,useHostTransitionStatus:um,useFormState:E0,useActionState:E0,useOptimistic:function(t,e){var n=Gt();return ut!==null?_y(n,ut,t,e):(n.baseState=t,[t,n.queue.dispatch])},useMemoCache:rm,useCacheRefresh:Fy};ky.useEffectEvent=by;function _d(t,e,n,i){e=t.memoizedState,n=n(i,e),n=n==null?e:Mt({},e,n),t.memoizedState=n,t.lanes===0&&(t.updateQueue.baseState=n)}var hp={enqueueSetState:function(t,e,n){t=t._reactInternals;var i=Zn(),s=Xs(i);s.payload=e,n!=null&&(s.callback=n),e=Ys(t,s,i),e!==null&&(Ln(e,t,i),qo(e,t,i))},enqueueReplaceState:function(t,e,n){t=t._reactInternals;var i=Zn(),s=Xs(i);s.tag=1,s.payload=e,n!=null&&(s.callback=n),e=Ys(t,s,i),e!==null&&(Ln(e,t,i),qo(e,t,i))},enqueueForceUpdate:function(t,e){t=t._reactInternals;var n=Zn(),i=Xs(n);i.tag=2,e!=null&&(i.callback=e),e=Ys(t,i,n),e!==null&&(Ln(e,t,n),qo(e,t,n))}};function b0(t,e,n,i,s,r,a){return t=t.stateNode,typeof t.shouldComponentUpdate=="function"?t.shouldComponentUpdate(i,r,a):e.prototype&&e.prototype.isPureReactComponent?!rl(n,i)||!rl(s,r):!0}function w0(t,e,n,i){t=e.state,typeof e.componentWillReceiveProps=="function"&&e.componentWillReceiveProps(n,i),typeof e.UNSAFE_componentWillReceiveProps=="function"&&e.UNSAFE_componentWillReceiveProps(n,i),e.state!==t&&hp.enqueueReplaceState(e,e.state,null)}function Gr(t,e){var n=e;if("ref"in e){n={};for(var i in e)i!=="ref"&&(n[i]=e[i])}if(t=t.defaultProps){n===e&&(n=Mt({},n));for(var s in t)n[s]===void 0&&(n[s]=t[s])}return n}function Wy(t){eu(t)}function Xy(t){console.error(t)}function Yy(t){eu(t)}function cu(t,e){try{var n=t.onUncaughtError;n(e.value,{componentStack:e.stack})}catch(i){setTimeout(function(){throw i})}}function C0(t,e,n){try{var i=t.onCaughtError;i(n.value,{componentStack:n.stack,errorBoundary:e.tag===1?e.stateNode:null})}catch(s){setTimeout(function(){throw s})}}function dp(t,e,n){return n=Xs(n),n.tag=3,n.payload={element:null},n.callback=function(){cu(t,e)},n}function qy(t){return t=Xs(t),t.tag=3,t}function Qy(t,e,n,i){var s=n.type.getDerivedStateFromError;if(typeof s=="function"){var r=i.value;t.payload=function(){return s(r)},t.callback=function(){C0(e,n,i)}}var a=n.stateNode;a!==null&&typeof a.componentDidCatch=="function"&&(t.callback=function(){C0(e,n,i),typeof s!="function"&&(qs===null?qs=new Set([this]):qs.add(this));var o=i.stack;this.componentDidCatch(i.value,{componentStack:o!==null?o:""})})}function $M(t,e,n,i,s){if(n.flags|=32768,i!==null&&typeof i=="object"&&typeof i.then=="function"){if(e=n.alternate,e!==null&&Ka(e,n,s,!0),n=Jn.current,n!==null){switch(n.tag){case 31:case 13:return hi===null?pu():n.alternate===null&&Lt===0&&(Lt=3),n.flags&=-257,n.flags|=65536,n.lanes=s,i===su?n.flags|=16384:(e=n.updateQueue,e===null?n.updateQueue=new Set([i]):e.add(i),Ud(t,i,s)),!1;case 22:return n.flags|=65536,i===su?n.flags|=16384:(e=n.updateQueue,e===null?(e={transitions:null,markerInstances:null,retryQueue:new Set([i])},n.updateQueue=e):(n=e.retryQueue,n===null?e.retryQueue=new Set([i]):n.add(i)),Ud(t,i,s)),!1}throw Error(J(435,n.tag))}return Ud(t,i,s),pu(),!1}if(qe)return e=Jn.current,e!==null?((e.flags&65536)===0&&(e.flags|=256),e.flags|=65536,e.lanes=s,i!==ep&&(t=Error(J(422),{cause:i}),ol(ui(t,n)))):(i!==ep&&(e=Error(J(423),{cause:i}),ol(ui(e,n))),t=t.current.alternate,t.flags|=65536,s&=-s,t.lanes|=s,i=ui(i,n),s=dp(t.stateNode,i,s),xd(t,s),Lt!==4&&(Lt=2)),!1;var r=Error(J(520),{cause:i});if(r=ui(r,n),$o===null?$o=[r]:$o.push(r),Lt!==4&&(Lt=2),e===null)return!0;i=ui(i,n),n=e;do{switch(n.tag){case 3:return n.flags|=65536,t=s&-s,n.lanes|=t,t=dp(n.stateNode,i,t),xd(n,t),!1;case 1:if(e=n.type,r=n.stateNode,(n.flags&128)===0&&(typeof e.getDerivedStateFromError=="function"||r!==null&&typeof r.componentDidCatch=="function"&&(qs===null||!qs.has(r))))return n.flags|=65536,s&=-s,n.lanes|=s,s=qy(s),Qy(s,t,n,i),xd(n,s),!1}n=n.return}while(n!==null);return!1}var dm=Error(J(461)),Qt=!1;function hn(t,e,n,i){e.child=t===null?ly(e,null,n,i):Fr(e,t.child,n,i)}function R0(t,e,n,i,s){n=n.render;var r=e.ref;if("ref"in i){var a={};for(var o in i)o!=="ref"&&(a[o]=i[o])}else a=i;return Or(e),i=tm(t,e,n,a,r,s),o=nm(),t!==null&&!Qt?(im(t,e,s),gs(t,e,s)):(qe&&o&&qp(e),e.flags|=1,hn(t,e,i,s),e.child)}function D0(t,e,n,i,s){if(t===null){var r=n.type;return typeof r=="function"&&!Yp(r)&&r.defaultProps===void 0&&n.compare===null?(e.tag=15,e.type=r,Zy(t,e,r,i,s)):(t=Fc(n.type,null,i,e,e.mode,s),t.ref=e.ref,t.return=e,e.child=t)}if(r=t.child,!pm(t,s)){var a=r.memoizedProps;if(n=n.compare,n=n!==null?n:rl,n(a,i)&&t.ref===e.ref)return gs(t,e,s)}return e.flags|=1,t=us(r,i),t.ref=e.ref,t.return=e,e.child=t}function Zy(t,e,n,i,s){if(t!==null){var r=t.memoizedProps;if(rl(r,i)&&t.ref===e.ref)if(Qt=!1,e.pendingProps=i=r,pm(t,s))(t.flags&131072)!==0&&(Qt=!0);else return e.lanes=t.lanes,gs(t,e,s)}return pp(t,e,n,i,s)}function Ky(t,e,n,i){var s=i.children,r=t!==null?t.memoizedState:null;if(t===null&&e.stateNode===null&&(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),i.mode==="hidden"){if((e.flags&128)!==0){if(r=r!==null?r.baseLanes|n:n,t!==null){for(i=e.child=t.child,s=0;i!==null;)s=s|i.lanes|i.childLanes,i=i.sibling;i=s&~r}else i=0,e.child=null;return U0(t,e,r,n,i)}if((n&536870912)!==0)e.memoizedState={baseLanes:0,cachePool:null},t!==null&&zc(e,r!==null?r.cachePool:null),r!==null?y0(e,r):op(),fy(e);else return i=e.lanes=536870912,U0(t,e,r!==null?r.baseLanes|n:n,n,i)}else r!==null?(zc(e,r.cachePool),y0(e,r),Ns(e),e.memoizedState=null):(t!==null&&zc(e,null),op(),Ns(e));return hn(t,e,s,n),e.child}function Ho(t,e){return t!==null&&t.tag===22||e.stateNode!==null||(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),e.sibling}function U0(t,e,n,i,s){var r=Kp();return r=r===null?null:{parent:qt._currentValue,pool:r},e.memoizedState={baseLanes:n,cachePool:r},t!==null&&zc(e,null),op(),fy(e),t!==null&&Ka(t,e,i,!0),e.childLanes=s,null}function Vc(t,e){return e=uu({mode:e.mode,children:e.children},t.mode),e.ref=t.ref,t.child=e,e.return=t,e}function B0(t,e,n){return Fr(e,t.child,null,n),t=Vc(e,e.pendingProps),t.flags|=2,kn(e),e.memoizedState=null,t}function eE(t,e,n){var i=e.pendingProps,s=(e.flags&128)!==0;if(e.flags&=-129,t===null){if(qe){if(i.mode==="hidden")return t=Vc(e,i),e.lanes=536870912,Ho(null,t);if(lp(e),(t=At)?(t=V_(t,fi),t=t!==null&&t.data==="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:js!==null?{id:Oi,overflow:Fi}:null,retryLane:536870912,hydrationErrors:null},n=ty(t),n.return=e,e.child=n,pn=e,At=null)):t=null,t===null)throw $s(e);return e.lanes=536870912,null}return Vc(e,i)}var r=t.memoizedState;if(r!==null){var a=r.dehydrated;if(lp(e),s)if(e.flags&256)e.flags&=-257,e=B0(t,e,n);else if(e.memoizedState!==null)e.child=t.child,e.flags|=128,e=null;else throw Error(J(558));else if(Qt||Ka(t,e,n,!1),s=(n&t.childLanes)!==0,Qt||s){if(i=pt,i!==null&&(a=wx(i,n),a!==0&&a!==r.retryLane))throw r.retryLane=a,Wr(t,a),Ln(i,t,a),dm;pu(),e=B0(t,e,n)}else t=r.treeContext,At=di(a.nextSibling),pn=e,qe=!0,Ws=null,fi=!1,t!==null&&iy(e,t),e=Vc(e,i),e.flags|=4096;return e}return t=us(t.child,{mode:i.mode,children:i.children}),t.ref=e.ref,e.child=t,t.return=e,t}function kc(t,e){var n=e.ref;if(n===null)t!==null&&t.ref!==null&&(e.flags|=4194816);else{if(typeof n!="function"&&typeof n!="object")throw Error(J(284));(t===null||t.ref!==n)&&(e.flags|=4194816)}}function pp(t,e,n,i,s){return Or(e),n=tm(t,e,n,i,void 0,s),i=nm(),t!==null&&!Qt?(im(t,e,s),gs(t,e,s)):(qe&&i&&qp(e),e.flags|=1,hn(t,e,n,s),e.child)}function I0(t,e,n,i,s,r){return Or(e),e.updateQueue=null,n=dy(e,i,n,s),hy(t),i=nm(),t!==null&&!Qt?(im(t,e,r),gs(t,e,r)):(qe&&i&&qp(e),e.flags|=1,hn(t,e,n,r),e.child)}function P0(t,e,n,i,s){if(Or(e),e.stateNode===null){var r=ba,a=n.contextType;typeof a=="object"&&a!==null&&(r=mn(a)),r=new n(i,r),e.memoizedState=r.state!==null&&r.state!==void 0?r.state:null,r.updater=hp,e.stateNode=r,r._reactInternals=e,r=e.stateNode,r.props=i,r.state=e.memoizedState,r.refs={},jp(e),a=n.contextType,r.context=typeof a=="object"&&a!==null?mn(a):ba,r.state=e.memoizedState,a=n.getDerivedStateFromProps,typeof a=="function"&&(_d(e,n,a,i),r.state=e.memoizedState),typeof n.getDerivedStateFromProps=="function"||typeof r.getSnapshotBeforeUpdate=="function"||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(a=r.state,typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount(),a!==r.state&&hp.enqueueReplaceState(r,r.state,null),Zo(e,i,r,s),Qo(),r.state=e.memoizedState),typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!0}else if(t===null){r=e.stateNode;var o=e.memoizedProps,l=Gr(n,o);r.props=l;var c=r.context,h=n.contextType;a=ba,typeof h=="object"&&h!==null&&(a=mn(h));var p=n.getDerivedStateFromProps;h=typeof p=="function"||typeof r.getSnapshotBeforeUpdate=="function",o=e.pendingProps!==o,h||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(o||c!==a)&&w0(e,r,i,a),Is=!1;var u=e.memoizedState;r.state=u,Zo(e,i,r,s),Qo(),c=e.memoizedState,o||u!==c||Is?(typeof p=="function"&&(_d(e,n,p,i),c=e.memoizedState),(l=Is||b0(e,n,l,i,u,c,a))?(h||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount()),typeof r.componentDidMount=="function"&&(e.flags|=4194308)):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),e.memoizedProps=i,e.memoizedState=c),r.props=i,r.state=c,r.context=a,i=l):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!1)}else{r=e.stateNode,rp(t,e),a=e.memoizedProps,h=Gr(n,a),r.props=h,p=e.pendingProps,u=r.context,c=n.contextType,l=ba,typeof c=="object"&&c!==null&&(l=mn(c)),o=n.getDerivedStateFromProps,(c=typeof o=="function"||typeof r.getSnapshotBeforeUpdate=="function")||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(a!==p||u!==l)&&w0(e,r,i,l),Is=!1,u=e.memoizedState,r.state=u,Zo(e,i,r,s),Qo();var d=e.memoizedState;a!==p||u!==d||Is||t!==null&&t.dependencies!==null&&iu(t.dependencies)?(typeof o=="function"&&(_d(e,n,o,i),d=e.memoizedState),(h=Is||b0(e,n,h,i,u,d,l)||t!==null&&t.dependencies!==null&&iu(t.dependencies))?(c||typeof r.UNSAFE_componentWillUpdate!="function"&&typeof r.componentWillUpdate!="function"||(typeof r.componentWillUpdate=="function"&&r.componentWillUpdate(i,d,l),typeof r.UNSAFE_componentWillUpdate=="function"&&r.UNSAFE_componentWillUpdate(i,d,l)),typeof r.componentDidUpdate=="function"&&(e.flags|=4),typeof r.getSnapshotBeforeUpdate=="function"&&(e.flags|=1024)):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),e.memoizedProps=i,e.memoizedState=d),r.props=i,r.state=d,r.context=l,i=h):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),i=!1)}return r=i,kc(t,e),i=(e.flags&128)!==0,r||i?(r=e.stateNode,n=i&&typeof n.getDerivedStateFromError!="function"?null:r.render(),e.flags|=1,t!==null&&i?(e.child=Fr(e,t.child,null,s),e.child=Fr(e,null,n,s)):hn(t,e,n,s),e.memoizedState=r.state,t=e.child):t=gs(t,e,s),t}function L0(t,e,n,i){return Nr(),e.flags|=256,hn(t,e,n,i),e.child}var Sd={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function Ad(t){return{baseLanes:t,cachePool:ry()}}function Md(t,e,n){return t=t!==null?t.childLanes&~n:0,e&&(t|=Xn),t}function Jy(t,e,n){var i=e.pendingProps,s=!1,r=(e.flags&128)!==0,a;if((a=r)||(a=t!==null&&t.memoizedState===null?!1:(zt.current&2)!==0),a&&(s=!0,e.flags&=-129),a=(e.flags&32)!==0,e.flags&=-33,t===null){if(qe){if(s?Ls(e):Ns(e),(t=At)?(t=V_(t,fi),t=t!==null&&t.data!=="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:js!==null?{id:Oi,overflow:Fi}:null,retryLane:536870912,hydrationErrors:null},n=ty(t),n.return=e,e.child=n,pn=e,At=null)):t=null,t===null)throw $s(e);return Cp(t)?e.lanes=32:e.lanes=536870912,null}var o=i.children;return i=i.fallback,s?(Ns(e),s=e.mode,o=uu({mode:"hidden",children:o},s),i=Br(i,s,n,null),o.return=e,i.return=e,o.sibling=i,e.child=o,i=e.child,i.memoizedState=Ad(n),i.childLanes=Md(t,a,n),e.memoizedState=Sd,Ho(null,i)):(Ls(e),mp(e,o))}var l=t.memoizedState;if(l!==null&&(o=l.dehydrated,o!==null)){if(r)e.flags&256?(Ls(e),e.flags&=-257,e=Ed(t,e,n)):e.memoizedState!==null?(Ns(e),e.child=t.child,e.flags|=128,e=null):(Ns(e),o=i.fallback,s=e.mode,i=uu({mode:"visible",children:i.children},s),o=Br(o,s,n,null),o.flags|=2,i.return=e,o.return=e,i.sibling=o,e.child=i,Fr(e,t.child,null,n),i=e.child,i.memoizedState=Ad(n),i.childLanes=Md(t,a,n),e.memoizedState=Sd,e=Ho(null,i));else if(Ls(e),Cp(o)){if(a=o.nextSibling&&o.nextSibling.dataset,a)var c=a.dgst;a=c,i=Error(J(419)),i.stack="",i.digest=a,ol({value:i,source:null,stack:null}),e=Ed(t,e,n)}else if(Qt||Ka(t,e,n,!1),a=(n&t.childLanes)!==0,Qt||a){if(a=pt,a!==null&&(i=wx(a,n),i!==0&&i!==l.retryLane))throw l.retryLane=i,Wr(t,i),Ln(a,t,i),dm;wp(o)||pu(),e=Ed(t,e,n)}else wp(o)?(e.flags|=192,e.child=t.child,e=null):(t=l.treeContext,At=di(o.nextSibling),pn=e,qe=!0,Ws=null,fi=!1,t!==null&&iy(e,t),e=mp(e,i.children),e.flags|=4096);return e}return s?(Ns(e),o=i.fallback,s=e.mode,l=t.child,c=l.sibling,i=us(l,{mode:"hidden",children:i.children}),i.subtreeFlags=l.subtreeFlags&65011712,c!==null?o=us(c,o):(o=Br(o,s,n,null),o.flags|=2),o.return=e,i.return=e,i.sibling=o,e.child=i,Ho(null,i),i=e.child,o=t.child.memoizedState,o===null?o=Ad(n):(s=o.cachePool,s!==null?(l=qt._currentValue,s=s.parent!==l?{parent:l,pool:l}:s):s=ry(),o={baseLanes:o.baseLanes|n,cachePool:s}),i.memoizedState=o,i.childLanes=Md(t,a,n),e.memoizedState=Sd,Ho(t.child,i)):(Ls(e),n=t.child,t=n.sibling,n=us(n,{mode:"visible",children:i.children}),n.return=e,n.sibling=null,t!==null&&(a=e.deletions,a===null?(e.deletions=[t],e.flags|=16):a.push(t)),e.child=n,e.memoizedState=null,n)}function mp(t,e){return e=uu({mode:"visible",children:e},t.mode),e.return=t,t.child=e}function uu(t,e){return t=Wn(22,t,null,e),t.lanes=0,t}function Ed(t,e,n){return Fr(e,t.child,null,n),t=mp(e,e.pendingProps.children),t.flags|=2,e.memoizedState=null,t}function N0(t,e,n){t.lanes|=e;var i=t.alternate;i!==null&&(i.lanes|=e),np(t.return,e,n)}function Td(t,e,n,i,s,r){var a=t.memoizedState;a===null?t.memoizedState={isBackwards:e,rendering:null,renderingStartTime:0,last:i,tail:n,tailMode:s,treeForkCount:r}:(a.isBackwards=e,a.rendering=null,a.renderingStartTime=0,a.last=i,a.tail=n,a.tailMode=s,a.treeForkCount=r)}function jy(t,e,n){var i=e.pendingProps,s=i.revealOrder,r=i.tail;i=i.children;var a=zt.current,o=(a&2)!==0;if(o?(a=a&1|2,e.flags|=128):a&=1,vt(zt,a),hn(t,e,i,n),i=qe?al:0,!o&&t!==null&&(t.flags&128)!==0)e:for(t=e.child;t!==null;){if(t.tag===13)t.memoizedState!==null&&N0(t,n,e);else if(t.tag===19)N0(t,n,e);else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break e;for(;t.sibling===null;){if(t.return===null||t.return===e)break e;t=t.return}t.sibling.return=t.return,t=t.sibling}switch(s){case"forwards":for(n=e.child,s=null;n!==null;)t=n.alternate,t!==null&&au(t)===null&&(s=n),n=n.sibling;n=s,n===null?(s=e.child,e.child=null):(s=n.sibling,n.sibling=null),Td(e,!1,s,n,r,i);break;case"backwards":case"unstable_legacy-backwards":for(n=null,s=e.child,e.child=null;s!==null;){if(t=s.alternate,t!==null&&au(t)===null){e.child=s;break}t=s.sibling,s.sibling=n,n=s,s=t}Td(e,!0,n,null,r,i);break;case"together":Td(e,!1,null,null,void 0,i);break;default:e.memoizedState=null}return e.child}function gs(t,e,n){if(t!==null&&(e.dependencies=t.dependencies),tr|=e.lanes,(n&e.childLanes)===0)if(t!==null){if(Ka(t,e,n,!1),(n&e.childLanes)===0)return null}else return null;if(t!==null&&e.child!==t.child)throw Error(J(153));if(e.child!==null){for(t=e.child,n=us(t,t.pendingProps),e.child=n,n.return=e;t.sibling!==null;)t=t.sibling,n=n.sibling=us(t,t.pendingProps),n.return=e;n.sibling=null}return e.child}function pm(t,e){return(t.lanes&e)!==0?!0:(t=t.dependencies,!!(t!==null&&iu(t)))}function tE(t,e,n){switch(e.tag){case 3:Kc(e,e.stateNode.containerInfo),Ps(e,qt,t.memoizedState.cache),Nr();break;case 27:case 5:kd(e);break;case 4:Kc(e,e.stateNode.containerInfo);break;case 10:Ps(e,e.type,e.memoizedProps.value);break;case 31:if(e.memoizedState!==null)return e.flags|=128,lp(e),null;break;case 13:var i=e.memoizedState;if(i!==null)return i.dehydrated!==null?(Ls(e),e.flags|=128,null):(n&e.child.childLanes)!==0?Jy(t,e,n):(Ls(e),t=gs(t,e,n),t!==null?t.sibling:null);Ls(e);break;case 19:var s=(t.flags&128)!==0;if(i=(n&e.childLanes)!==0,i||(Ka(t,e,n,!1),i=(n&e.childLanes)!==0),s){if(i)return jy(t,e,n);e.flags|=128}if(s=e.memoizedState,s!==null&&(s.rendering=null,s.tail=null,s.lastEffect=null),vt(zt,zt.current),i)break;return null;case 22:return e.lanes=0,Ky(t,e,n,e.pendingProps);case 24:Ps(e,qt,t.memoizedState.cache)}return gs(t,e,n)}function $y(t,e,n){if(t!==null)if(t.memoizedProps!==e.pendingProps)Qt=!0;else{if(!pm(t,n)&&(e.flags&128)===0)return Qt=!1,tE(t,e,n);Qt=(t.flags&131072)!==0}else Qt=!1,qe&&(e.flags&1048576)!==0&&ny(e,al,e.index);switch(e.lanes=0,e.tag){case 16:e:{var i=e.pendingProps;if(t=Rr(e.elementType),e.type=t,typeof t=="function")Yp(t)?(i=Gr(t,i),e.tag=1,e=P0(null,e,t,i,n)):(e.tag=0,e=pp(null,e,t,i,n));else{if(t!=null){var s=t.$$typeof;if(s===Bp){e.tag=11,e=R0(null,e,t,i,n);break e}else if(s===Ip){e.tag=14,e=D0(null,e,t,i,n);break e}}throw e=Hd(t)||t,Error(J(306,e,""))}}return e;case 0:return pp(t,e,e.type,e.pendingProps,n);case 1:return i=e.type,s=Gr(i,e.pendingProps),P0(t,e,i,s,n);case 3:e:{if(Kc(e,e.stateNode.containerInfo),t===null)throw Error(J(387));i=e.pendingProps;var r=e.memoizedState;s=r.element,rp(t,e),Zo(e,i,null,n);var a=e.memoizedState;if(i=a.cache,Ps(e,qt,i),i!==r.cache&&ip(e,[qt],n,!0),Qo(),i=a.element,r.isDehydrated)if(r={element:i,isDehydrated:!1,cache:a.cache},e.updateQueue.baseState=r,e.memoizedState=r,e.flags&256){e=L0(t,e,i,n);break e}else if(i!==s){s=ui(Error(J(424)),e),ol(s),e=L0(t,e,i,n);break e}else for(t=e.stateNode.containerInfo,t.nodeType===9?t=t.body:t=t.nodeName==="HTML"?t.ownerDocument.body:t,At=di(t.firstChild),pn=e,qe=!0,Ws=null,fi=!0,n=ly(e,null,i,n),e.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Nr(),i===s){e=gs(t,e,n);break e}hn(t,e,i,n)}e=e.child}return e;case 26:return kc(t,e),t===null?(n=sx(e.type,null,e.pendingProps,null))?e.memoizedState=n:qe||(n=e.type,t=e.pendingProps,i=xu(ks.current).createElement(n),i[dn]=e,i[Nn]=t,gn(i,n,t),on(i),e.stateNode=i):e.memoizedState=sx(e.type,t.memoizedProps,e.pendingProps,t.memoizedState),null;case 27:return kd(e),t===null&&qe&&(i=e.stateNode=k_(e.type,e.pendingProps,ks.current),pn=e,fi=!0,s=At,ir(e.type)?(Rp=s,At=di(i.firstChild)):At=s),hn(t,e,e.pendingProps.children,n),kc(t,e),t===null&&(e.flags|=4194304),e.child;case 5:return t===null&&qe&&((s=i=At)&&(i=RE(i,e.type,e.pendingProps,fi),i!==null?(e.stateNode=i,pn=e,At=di(i.firstChild),fi=!1,s=!0):s=!1),s||$s(e)),kd(e),s=e.type,r=e.pendingProps,a=t!==null?t.memoizedProps:null,i=r.children,Tp(s,r)?i=null:a!==null&&Tp(s,a)&&(e.flags|=32),e.memoizedState!==null&&(s=tm(t,e,YM,null,null,n),dl._currentValue=s),kc(t,e),hn(t,e,i,n),e.child;case 6:return t===null&&qe&&((t=n=At)&&(n=DE(n,e.pendingProps,fi),n!==null?(e.stateNode=n,pn=e,At=null,t=!0):t=!1),t||$s(e)),null;case 13:return Jy(t,e,n);case 4:return Kc(e,e.stateNode.containerInfo),i=e.pendingProps,t===null?e.child=Fr(e,null,i,n):hn(t,e,i,n),e.child;case 11:return R0(t,e,e.type,e.pendingProps,n);case 7:return hn(t,e,e.pendingProps,n),e.child;case 8:return hn(t,e,e.pendingProps.children,n),e.child;case 12:return hn(t,e,e.pendingProps.children,n),e.child;case 10:return i=e.pendingProps,Ps(e,e.type,i.value),hn(t,e,i.children,n),e.child;case 9:return s=e.type._context,i=e.pendingProps.children,Or(e),s=mn(s),i=i(s),e.flags|=1,hn(t,e,i,n),e.child;case 14:return D0(t,e,e.type,e.pendingProps,n);case 15:return Zy(t,e,e.type,e.pendingProps,n);case 19:return jy(t,e,n);case 31:return eE(t,e,n);case 22:return Ky(t,e,n,e.pendingProps);case 24:return Or(e),i=mn(qt),t===null?(s=Kp(),s===null&&(s=pt,r=Zp(),s.pooledCache=r,r.refCount++,r!==null&&(s.pooledCacheLanes|=n),s=r),e.memoizedState={parent:i,cache:s},jp(e),Ps(e,qt,s)):((t.lanes&n)!==0&&(rp(t,e),Zo(e,null,null,n),Qo()),s=t.memoizedState,r=e.memoizedState,s.parent!==i?(s={parent:i,cache:i},e.memoizedState=s,e.lanes===0&&(e.memoizedState=e.updateQueue.baseState=s),Ps(e,qt,i)):(i=r.cache,Ps(e,qt,i),i!==s.cache&&ip(e,[qt],n,!0))),hn(t,e,e.pendingProps.children,n),e.child;case 29:throw e.pendingProps}throw Error(J(156,e.tag))}function ts(t){t.flags|=4}function bd(t,e,n,i,s){if((e=(t.mode&32)!==0)&&(e=!1),e){if(t.flags|=16777216,(s&335544128)===s)if(t.stateNode.complete)t.flags|=8192;else if(M_())t.flags|=8192;else throw Pr=su,Jp}else t.flags&=-16777217}function O0(t,e){if(e.type!=="stylesheet"||(e.state.loading&4)!==0)t.flags&=-16777217;else if(t.flags|=16777216,!Y_(e))if(M_())t.flags|=8192;else throw Pr=su,Jp}function wc(t,e){e!==null&&(t.flags|=4),t.flags&16384&&(e=t.tag!==22?Ex():536870912,t.lanes|=e,ka|=e)}function Po(t,e){if(!qe)switch(t.tailMode){case"hidden":e=t.tail;for(var n=null;e!==null;)e.alternate!==null&&(n=e),e=e.sibling;n===null?t.tail=null:n.sibling=null;break;case"collapsed":n=t.tail;for(var i=null;n!==null;)n.alternate!==null&&(i=n),n=n.sibling;i===null?e||t.tail===null?t.tail=null:t.tail.sibling=null:i.sibling=null}}function St(t){var e=t.alternate!==null&&t.alternate.child===t.child,n=0,i=0;if(e)for(var s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags&65011712,i|=s.flags&65011712,s.return=t,s=s.sibling;else for(s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags,i|=s.flags,s.return=t,s=s.sibling;return t.subtreeFlags|=i,t.childLanes=n,e}function nE(t,e,n){var i=e.pendingProps;switch(Qp(e),e.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return St(e),null;case 1:return St(e),null;case 3:return n=e.stateNode,i=null,t!==null&&(i=t.memoizedState.cache),e.memoizedState.cache!==i&&(e.flags|=2048),fs(qt),Oa(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(t===null||t.child===null)&&(pa(e)?ts(e):t===null||t.memoizedState.isDehydrated&&(e.flags&256)===0||(e.flags|=1024,vd())),St(e),null;case 26:var s=e.type,r=e.memoizedState;return t===null?(ts(e),r!==null?(St(e),O0(e,r)):(St(e),bd(e,s,null,i,n))):r?r!==t.memoizedState?(ts(e),St(e),O0(e,r)):(St(e),e.flags&=-16777217):(t=t.memoizedProps,t!==i&&ts(e),St(e),bd(e,s,t,i,n)),null;case 27:if(Jc(e),n=ks.current,s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&ts(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return St(e),null}t=Gi.current,pa(e)?h0(e,t):(t=k_(s,i,n),e.stateNode=t,ts(e))}return St(e),null;case 5:if(Jc(e),s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&ts(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return St(e),null}if(r=Gi.current,pa(e))h0(e,r);else{var a=xu(ks.current);switch(r){case 1:r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case 2:r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;default:switch(s){case"svg":r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case"math":r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;case"script":r=a.createElement("div"),r.innerHTML="<script><\/script>",r=r.removeChild(r.firstChild);break;case"select":r=typeof i.is=="string"?a.createElement("select",{is:i.is}):a.createElement("select"),i.multiple?r.multiple=!0:i.size&&(r.size=i.size);break;default:r=typeof i.is=="string"?a.createElement(s,{is:i.is}):a.createElement(s)}}r[dn]=e,r[Nn]=i;e:for(a=e.child;a!==null;){if(a.tag===5||a.tag===6)r.appendChild(a.stateNode);else if(a.tag!==4&&a.tag!==27&&a.child!==null){a.child.return=a,a=a.child;continue}if(a===e)break e;for(;a.sibling===null;){if(a.return===null||a.return===e)break e;a=a.return}a.sibling.return=a.return,a=a.sibling}e.stateNode=r;e:switch(gn(r,s,i),s){case"button":case"input":case"select":case"textarea":i=!!i.autoFocus;break e;case"img":i=!0;break e;default:i=!1}i&&ts(e)}}return St(e),bd(e,e.type,t===null?null:t.memoizedProps,e.pendingProps,n),null;case 6:if(t&&e.stateNode!=null)t.memoizedProps!==i&&ts(e);else{if(typeof i!="string"&&e.stateNode===null)throw Error(J(166));if(t=ks.current,pa(e)){if(t=e.stateNode,n=e.memoizedProps,i=null,s=pn,s!==null)switch(s.tag){case 27:case 5:i=s.memoizedProps}t[dn]=e,t=!!(t.nodeValue===n||i!==null&&i.suppressHydrationWarning===!0||z_(t.nodeValue,n)),t||$s(e,!0)}else t=xu(t).createTextNode(i),t[dn]=e,e.stateNode=t}return St(e),null;case 31:if(n=e.memoizedState,t===null||t.memoizedState!==null){if(i=pa(e),n!==null){if(t===null){if(!i)throw Error(J(318));if(t=e.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(557));t[dn]=e}else Nr(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;St(e),t=!1}else n=vd(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=n),t=!0;if(!t)return e.flags&256?(kn(e),e):(kn(e),null);if((e.flags&128)!==0)throw Error(J(558))}return St(e),null;case 13:if(i=e.memoizedState,t===null||t.memoizedState!==null&&t.memoizedState.dehydrated!==null){if(s=pa(e),i!==null&&i.dehydrated!==null){if(t===null){if(!s)throw Error(J(318));if(s=e.memoizedState,s=s!==null?s.dehydrated:null,!s)throw Error(J(317));s[dn]=e}else Nr(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;St(e),s=!1}else s=vd(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=s),s=!0;if(!s)return e.flags&256?(kn(e),e):(kn(e),null)}return kn(e),(e.flags&128)!==0?(e.lanes=n,e):(n=i!==null,t=t!==null&&t.memoizedState!==null,n&&(i=e.child,s=null,i.alternate!==null&&i.alternate.memoizedState!==null&&i.alternate.memoizedState.cachePool!==null&&(s=i.alternate.memoizedState.cachePool.pool),r=null,i.memoizedState!==null&&i.memoizedState.cachePool!==null&&(r=i.memoizedState.cachePool.pool),r!==s&&(i.flags|=2048)),n!==t&&n&&(e.child.flags|=8192),wc(e,e.updateQueue),St(e),null);case 4:return Oa(),t===null&&Sm(e.stateNode.containerInfo),St(e),null;case 10:return fs(e.type),St(e),null;case 19:if(ln(zt),i=e.memoizedState,i===null)return St(e),null;if(s=(e.flags&128)!==0,r=i.rendering,r===null)if(s)Po(i,!1);else{if(Lt!==0||t!==null&&(t.flags&128)!==0)for(t=e.child;t!==null;){if(r=au(t),r!==null){for(e.flags|=128,Po(i,!1),t=r.updateQueue,e.updateQueue=t,wc(e,t),e.subtreeFlags=0,t=n,n=e.child;n!==null;)ey(n,t),n=n.sibling;return vt(zt,zt.current&1|2),qe&&rs(e,i.treeForkCount),e.child}t=t.sibling}i.tail!==null&&Yn()>hu&&(e.flags|=128,s=!0,Po(i,!1),e.lanes=4194304)}else{if(!s)if(t=au(r),t!==null){if(e.flags|=128,s=!0,t=t.updateQueue,e.updateQueue=t,wc(e,t),Po(i,!0),i.tail===null&&i.tailMode==="hidden"&&!r.alternate&&!qe)return St(e),null}else 2*Yn()-i.renderingStartTime>hu&&n!==536870912&&(e.flags|=128,s=!0,Po(i,!1),e.lanes=4194304);i.isBackwards?(r.sibling=e.child,e.child=r):(t=i.last,t!==null?t.sibling=r:e.child=r,i.last=r)}return i.tail!==null?(t=i.tail,i.rendering=t,i.tail=t.sibling,i.renderingStartTime=Yn(),t.sibling=null,n=zt.current,vt(zt,s?n&1|2:n&1),qe&&rs(e,i.treeForkCount),t):(St(e),null);case 22:case 23:return kn(e),$p(),i=e.memoizedState!==null,t!==null?t.memoizedState!==null!==i&&(e.flags|=8192):i&&(e.flags|=8192),i?(n&536870912)!==0&&(e.flags&128)===0&&(St(e),e.subtreeFlags&6&&(e.flags|=8192)):St(e),n=e.updateQueue,n!==null&&wc(e,n.retryQueue),n=null,t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),i=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(i=e.memoizedState.cachePool.pool),i!==n&&(e.flags|=2048),t!==null&&ln(Ir),null;case 24:return n=null,t!==null&&(n=t.memoizedState.cache),e.memoizedState.cache!==n&&(e.flags|=2048),fs(qt),St(e),null;case 25:return null;case 30:return null}throw Error(J(156,e.tag))}function iE(t,e){switch(Qp(e),e.tag){case 1:return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 3:return fs(qt),Oa(),t=e.flags,(t&65536)!==0&&(t&128)===0?(e.flags=t&-65537|128,e):null;case 26:case 27:case 5:return Jc(e),null;case 31:if(e.memoizedState!==null){if(kn(e),e.alternate===null)throw Error(J(340));Nr()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 13:if(kn(e),t=e.memoizedState,t!==null&&t.dehydrated!==null){if(e.alternate===null)throw Error(J(340));Nr()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 19:return ln(zt),null;case 4:return Oa(),null;case 10:return fs(e.type),null;case 22:case 23:return kn(e),$p(),t!==null&&ln(Ir),t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 24:return fs(qt),null;case 25:return null;default:return null}}function e_(t,e){switch(Qp(e),e.tag){case 3:fs(qt),Oa();break;case 26:case 27:case 5:Jc(e);break;case 4:Oa();break;case 31:e.memoizedState!==null&&kn(e);break;case 13:kn(e);break;case 19:ln(zt);break;case 10:fs(e.type);break;case 22:case 23:kn(e),$p(),t!==null&&ln(Ir);break;case 24:fs(qt)}}function El(t,e){try{var n=e.updateQueue,i=n!==null?n.lastEffect:null;if(i!==null){var s=i.next;n=s;do{if((n.tag&t)===t){i=void 0;var r=n.create,a=n.inst;i=r(),a.destroy=i}n=n.next}while(n!==s)}}catch(o){ot(e,e.return,o)}}function er(t,e,n){try{var i=e.updateQueue,s=i!==null?i.lastEffect:null;if(s!==null){var r=s.next;i=r;do{if((i.tag&t)===t){var a=i.inst,o=a.destroy;if(o!==void 0){a.destroy=void 0,s=e;var l=n,c=o;try{c()}catch(h){ot(s,l,h)}}}i=i.next}while(i!==r)}}catch(h){ot(e,e.return,h)}}function t_(t){var e=t.updateQueue;if(e!==null){var n=t.stateNode;try{uy(e,n)}catch(i){ot(t,t.return,i)}}}function n_(t,e,n){n.props=Gr(t.type,t.memoizedProps),n.state=t.memoizedState;try{n.componentWillUnmount()}catch(i){ot(t,e,i)}}function Jo(t,e){try{var n=t.ref;if(n!==null){switch(t.tag){case 26:case 27:case 5:var i=t.stateNode;break;case 30:i=t.stateNode;break;default:i=t.stateNode}typeof n=="function"?t.refCleanup=n(i):n.current=i}}catch(s){ot(t,e,s)}}function zi(t,e){var n=t.ref,i=t.refCleanup;if(n!==null)if(typeof i=="function")try{i()}catch(s){ot(t,e,s)}finally{t.refCleanup=null,t=t.alternate,t!=null&&(t.refCleanup=null)}else if(typeof n=="function")try{n(null)}catch(s){ot(t,e,s)}else n.current=null}function i_(t){var e=t.type,n=t.memoizedProps,i=t.stateNode;try{e:switch(e){case"button":case"input":case"select":case"textarea":n.autoFocus&&i.focus();break e;case"img":n.src?i.src=n.src:n.srcSet&&(i.srcset=n.srcSet)}}catch(s){ot(t,t.return,s)}}function wd(t,e,n){try{var i=t.stateNode;ME(i,t.type,n,e),i[Nn]=e}catch(s){ot(t,t.return,s)}}function s_(t){return t.tag===5||t.tag===3||t.tag===26||t.tag===27&&ir(t.type)||t.tag===4}function Cd(t){e:for(;;){for(;t.sibling===null;){if(t.return===null||s_(t.return))return null;t=t.return}for(t.sibling.return=t.return,t=t.sibling;t.tag!==5&&t.tag!==6&&t.tag!==18;){if(t.tag===27&&ir(t.type)||t.flags&2||t.child===null||t.tag===4)continue e;t.child.return=t,t=t.child}if(!(t.flags&2))return t.stateNode}}function gp(t,e,n){var i=t.tag;if(i===5||i===6)t=t.stateNode,e?(n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n).insertBefore(t,e):(e=n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n,e.appendChild(t),n=n._reactRootContainer,n!=null||e.onclick!==null||(e.onclick=ls));else if(i!==4&&(i===27&&ir(t.type)&&(n=t.stateNode,e=null),t=t.child,t!==null))for(gp(t,e,n),t=t.sibling;t!==null;)gp(t,e,n),t=t.sibling}function fu(t,e,n){var i=t.tag;if(i===5||i===6)t=t.stateNode,e?n.insertBefore(t,e):n.appendChild(t);else if(i!==4&&(i===27&&ir(t.type)&&(n=t.stateNode),t=t.child,t!==null))for(fu(t,e,n),t=t.sibling;t!==null;)fu(t,e,n),t=t.sibling}function r_(t){var e=t.stateNode,n=t.memoizedProps;try{for(var i=t.type,s=e.attributes;s.length;)e.removeAttributeNode(s[0]);gn(e,i,n),e[dn]=t,e[Nn]=n}catch(r){ot(t,t.return,r)}}var as=!1,Yt=!1,Rd=!1,F0=typeof WeakSet=="function"?WeakSet:Set,an=null;function sE(t,e){if(t=t.containerInfo,Mp=Au,t=Yx(t),kp(t)){if("selectionStart"in t)var n={start:t.selectionStart,end:t.selectionEnd};else e:{n=(n=t.ownerDocument)&&n.defaultView||window;var i=n.getSelection&&n.getSelection();if(i&&i.rangeCount!==0){n=i.anchorNode;var s=i.anchorOffset,r=i.focusNode;i=i.focusOffset;try{n.nodeType,r.nodeType}catch{n=null;break e}var a=0,o=-1,l=-1,c=0,h=0,p=t,u=null;t:for(;;){for(var d;p!==n||s!==0&&p.nodeType!==3||(o=a+s),p!==r||i!==0&&p.nodeType!==3||(l=a+i),p.nodeType===3&&(a+=p.nodeValue.length),(d=p.firstChild)!==null;)u=p,p=d;for(;;){if(p===t)break t;if(u===n&&++c===s&&(o=a),u===r&&++h===i&&(l=a),(d=p.nextSibling)!==null)break;p=u,u=p.parentNode}p=d}n=o===-1||l===-1?null:{start:o,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(Ep={focusedElem:t,selectionRange:n},Au=!1,an=e;an!==null;)if(e=an,t=e.child,(e.subtreeFlags&1028)!==0&&t!==null)t.return=e,an=t;else for(;an!==null;){switch(e=an,r=e.alternate,t=e.flags,e.tag){case 0:if((t&4)!==0&&(t=e.updateQueue,t=t!==null?t.events:null,t!==null))for(n=0;n<t.length;n++)s=t[n],s.ref.impl=s.nextImpl;break;case 11:case 15:break;case 1:if((t&1024)!==0&&r!==null){t=void 0,n=e,s=r.memoizedProps,r=r.memoizedState,i=n.stateNode;try{var v=Gr(n.type,s);t=i.getSnapshotBeforeUpdate(v,r),i.__reactInternalSnapshotBeforeUpdate=t}catch(M){ot(n,n.return,M)}}break;case 3:if((t&1024)!==0){if(t=e.stateNode.containerInfo,n=t.nodeType,n===9)bp(t);else if(n===1)switch(t.nodeName){case"HEAD":case"HTML":case"BODY":bp(t);break;default:t.textContent=""}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;default:if((t&1024)!==0)throw Error(J(163))}if(t=e.sibling,t!==null){t.return=e.return,an=t;break}an=e.return}}function a_(t,e,n){var i=n.flags;switch(n.tag){case 0:case 11:case 15:is(t,n),i&4&&El(5,n);break;case 1:if(is(t,n),i&4)if(t=n.stateNode,e===null)try{t.componentDidMount()}catch(a){ot(n,n.return,a)}else{var s=Gr(n.type,e.memoizedProps);e=e.memoizedState;try{t.componentDidUpdate(s,e,t.__reactInternalSnapshotBeforeUpdate)}catch(a){ot(n,n.return,a)}}i&64&&t_(n),i&512&&Jo(n,n.return);break;case 3:if(is(t,n),i&64&&(t=n.updateQueue,t!==null)){if(e=null,n.child!==null)switch(n.child.tag){case 27:case 5:e=n.child.stateNode;break;case 1:e=n.child.stateNode}try{uy(t,e)}catch(a){ot(n,n.return,a)}}break;case 27:e===null&&i&4&&r_(n);case 26:case 5:is(t,n),e===null&&i&4&&i_(n),i&512&&Jo(n,n.return);break;case 12:is(t,n);break;case 31:is(t,n),i&4&&c_(t,n);break;case 13:is(t,n),i&4&&u_(t,n),i&64&&(t=n.memoizedState,t!==null&&(t=t.dehydrated,t!==null&&(n=dE.bind(null,n),UE(t,n))));break;case 22:if(i=n.memoizedState!==null||as,!i){e=e!==null&&e.memoizedState!==null||Yt,s=as;var r=Yt;as=i,(Yt=e)&&!r?ss(t,n,(n.subtreeFlags&8772)!==0):is(t,n),as=s,Yt=r}break;case 30:break;default:is(t,n)}}function o_(t){var e=t.alternate;e!==null&&(t.alternate=null,o_(e)),t.child=null,t.deletions=null,t.sibling=null,t.tag===5&&(e=t.stateNode,e!==null&&Op(e)),t.stateNode=null,t.return=null,t.dependencies=null,t.memoizedProps=null,t.memoizedState=null,t.pendingProps=null,t.stateNode=null,t.updateQueue=null}var wt=null,In=!1;function ns(t,e,n){for(n=n.child;n!==null;)l_(t,e,n),n=n.sibling}function l_(t,e,n){if(qn&&typeof qn.onCommitFiberUnmount=="function")try{qn.onCommitFiberUnmount(vl,n)}catch{}switch(n.tag){case 26:Yt||zi(n,e),ns(t,e,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:Yt||zi(n,e);var i=wt,s=In;ir(n.type)&&(wt=n.stateNode,In=!1),ns(t,e,n),tl(n.stateNode),wt=i,In=s;break;case 5:Yt||zi(n,e);case 6:if(i=wt,s=In,wt=null,ns(t,e,n),wt=i,In=s,wt!==null)if(In)try{(wt.nodeType===9?wt.body:wt.nodeName==="HTML"?wt.ownerDocument.body:wt).removeChild(n.stateNode)}catch(r){ot(n,e,r)}else try{wt.removeChild(n.stateNode)}catch(r){ot(n,e,r)}break;case 18:wt!==null&&(In?(t=wt,$0(t.nodeType===9?t.body:t.nodeName==="HTML"?t.ownerDocument.body:t,n.stateNode),qa(t)):$0(wt,n.stateNode));break;case 4:i=wt,s=In,wt=n.stateNode.containerInfo,In=!0,ns(t,e,n),wt=i,In=s;break;case 0:case 11:case 14:case 15:er(2,n,e),Yt||er(4,n,e),ns(t,e,n);break;case 1:Yt||(zi(n,e),i=n.stateNode,typeof i.componentWillUnmount=="function"&&n_(n,e,i)),ns(t,e,n);break;case 21:ns(t,e,n);break;case 22:Yt=(i=Yt)||n.memoizedState!==null,ns(t,e,n),Yt=i;break;default:ns(t,e,n)}}function c_(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null))){t=t.dehydrated;try{qa(t)}catch(n){ot(e,e.return,n)}}}function u_(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null&&(t=t.dehydrated,t!==null))))try{qa(t)}catch(n){ot(e,e.return,n)}}function rE(t){switch(t.tag){case 31:case 13:case 19:var e=t.stateNode;return e===null&&(e=t.stateNode=new F0),e;case 22:return t=t.stateNode,e=t._retryCache,e===null&&(e=t._retryCache=new F0),e;default:throw Error(J(435,t.tag))}}function Cc(t,e){var n=rE(t);e.forEach(function(i){if(!n.has(i)){n.add(i);var s=pE.bind(null,t,i);i.then(s,s)}})}function Un(t,e){var n=e.deletions;if(n!==null)for(var i=0;i<n.length;i++){var s=n[i],r=t,a=e,o=a;e:for(;o!==null;){switch(o.tag){case 27:if(ir(o.type)){wt=o.stateNode,In=!1;break e}break;case 5:wt=o.stateNode,In=!1;break e;case 3:case 4:wt=o.stateNode.containerInfo,In=!0;break e}o=o.return}if(wt===null)throw Error(J(160));l_(r,a,s),wt=null,In=!1,r=s.alternate,r!==null&&(r.return=null),s.return=null}if(e.subtreeFlags&13886)for(e=e.child;e!==null;)f_(e,t),e=e.sibling}var Si=null;function f_(t,e){var n=t.alternate,i=t.flags;switch(t.tag){case 0:case 11:case 14:case 15:Un(e,t),Bn(t),i&4&&(er(3,t,t.return),El(3,t),er(5,t,t.return));break;case 1:Un(e,t),Bn(t),i&512&&(Yt||n===null||zi(n,n.return)),i&64&&as&&(t=t.updateQueue,t!==null&&(i=t.callbacks,i!==null&&(n=t.shared.hiddenCallbacks,t.shared.hiddenCallbacks=n===null?i:n.concat(i))));break;case 26:var s=Si;if(Un(e,t),Bn(t),i&512&&(Yt||n===null||zi(n,n.return)),i&4){var r=n!==null?n.memoizedState:null;if(i=t.memoizedState,n===null)if(i===null)if(t.stateNode===null){e:{i=t.type,n=t.memoizedProps,s=s.ownerDocument||s;t:switch(i){case"title":r=s.getElementsByTagName("title")[0],(!r||r[_l]||r[dn]||r.namespaceURI==="http://www.w3.org/2000/svg"||r.hasAttribute("itemprop"))&&(r=s.createElement(i),s.head.insertBefore(r,s.querySelector("head > title"))),gn(r,i,n),r[dn]=t,on(r),i=r;break e;case"link":var a=ax("link","href",s).get(i+(n.href||""));if(a){for(var o=0;o<a.length;o++)if(r=a[o],r.getAttribute("href")===(n.href==null||n.href===""?null:n.href)&&r.getAttribute("rel")===(n.rel==null?null:n.rel)&&r.getAttribute("title")===(n.title==null?null:n.title)&&r.getAttribute("crossorigin")===(n.crossOrigin==null?null:n.crossOrigin)){a.splice(o,1);break t}}r=s.createElement(i),gn(r,i,n),s.head.appendChild(r);break;case"meta":if(a=ax("meta","content",s).get(i+(n.content||""))){for(o=0;o<a.length;o++)if(r=a[o],r.getAttribute("content")===(n.content==null?null:""+n.content)&&r.getAttribute("name")===(n.name==null?null:n.name)&&r.getAttribute("property")===(n.property==null?null:n.property)&&r.getAttribute("http-equiv")===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute("charset")===(n.charSet==null?null:n.charSet)){a.splice(o,1);break t}}r=s.createElement(i),gn(r,i,n),s.head.appendChild(r);break;default:throw Error(J(468,i))}r[dn]=t,on(r),i=r}t.stateNode=i}else ox(s,t.type,t.stateNode);else t.stateNode=rx(s,i,t.memoizedProps);else r!==i?(r===null?n.stateNode!==null&&(n=n.stateNode,n.parentNode.removeChild(n)):r.count--,i===null?ox(s,t.type,t.stateNode):rx(s,i,t.memoizedProps)):i===null&&t.stateNode!==null&&wd(t,t.memoizedProps,n.memoizedProps)}break;case 27:Un(e,t),Bn(t),i&512&&(Yt||n===null||zi(n,n.return)),n!==null&&i&4&&wd(t,t.memoizedProps,n.memoizedProps);break;case 5:if(Un(e,t),Bn(t),i&512&&(Yt||n===null||zi(n,n.return)),t.flags&32){s=t.stateNode;try{za(s,"")}catch(v){ot(t,t.return,v)}}i&4&&t.stateNode!=null&&(s=t.memoizedProps,wd(t,s,n!==null?n.memoizedProps:s)),i&1024&&(Rd=!0);break;case 6:if(Un(e,t),Bn(t),i&4){if(t.stateNode===null)throw Error(J(162));i=t.memoizedProps,n=t.stateNode;try{n.nodeValue=i}catch(v){ot(t,t.return,v)}}break;case 3:if(Yc=null,s=Si,Si=yu(e.containerInfo),Un(e,t),Si=s,Bn(t),i&4&&n!==null&&n.memoizedState.isDehydrated)try{qa(e.containerInfo)}catch(v){ot(t,t.return,v)}Rd&&(Rd=!1,h_(t));break;case 4:i=Si,Si=yu(t.stateNode.containerInfo),Un(e,t),Bn(t),Si=i;break;case 12:Un(e,t),Bn(t);break;case 31:Un(e,t),Bn(t),i&4&&(i=t.updateQueue,i!==null&&(t.updateQueue=null,Cc(t,i)));break;case 13:Un(e,t),Bn(t),t.child.flags&8192&&t.memoizedState!==null!=(n!==null&&n.memoizedState!==null)&&(Lu=Yn()),i&4&&(i=t.updateQueue,i!==null&&(t.updateQueue=null,Cc(t,i)));break;case 22:s=t.memoizedState!==null;var l=n!==null&&n.memoizedState!==null,c=as,h=Yt;if(as=c||s,Yt=h||l,Un(e,t),Yt=h,as=c,Bn(t),i&8192)e:for(e=t.stateNode,e._visibility=s?e._visibility&-2:e._visibility|1,s&&(n===null||l||as||Yt||Dr(t)),n=null,e=t;;){if(e.tag===5||e.tag===26){if(n===null){l=n=e;try{if(r=l.stateNode,s)a=r.style,typeof a.setProperty=="function"?a.setProperty("display","none","important"):a.display="none";else{o=l.stateNode;var p=l.memoizedProps.style,u=p!=null&&p.hasOwnProperty("display")?p.display:null;o.style.display=u==null||typeof u=="boolean"?"":(""+u).trim()}}catch(v){ot(l,l.return,v)}}}else if(e.tag===6){if(n===null){l=e;try{l.stateNode.nodeValue=s?"":l.memoizedProps}catch(v){ot(l,l.return,v)}}}else if(e.tag===18){if(n===null){l=e;try{var d=l.stateNode;s?ex(d,!0):ex(l.stateNode,!1)}catch(v){ot(l,l.return,v)}}}else if((e.tag!==22&&e.tag!==23||e.memoizedState===null||e===t)&&e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;n===e&&(n=null),e=e.return}n===e&&(n=null),e.sibling.return=e.return,e=e.sibling}i&4&&(i=t.updateQueue,i!==null&&(n=i.retryQueue,n!==null&&(i.retryQueue=null,Cc(t,n))));break;case 19:Un(e,t),Bn(t),i&4&&(i=t.updateQueue,i!==null&&(t.updateQueue=null,Cc(t,i)));break;case 30:break;case 21:break;default:Un(e,t),Bn(t)}}function Bn(t){var e=t.flags;if(e&2){try{for(var n,i=t.return;i!==null;){if(s_(i)){n=i;break}i=i.return}if(n==null)throw Error(J(160));switch(n.tag){case 27:var s=n.stateNode,r=Cd(t);fu(t,r,s);break;case 5:var a=n.stateNode;n.flags&32&&(za(a,""),n.flags&=-33);var o=Cd(t);fu(t,o,a);break;case 3:case 4:var l=n.stateNode.containerInfo,c=Cd(t);gp(t,c,l);break;default:throw Error(J(161))}}catch(h){ot(t,t.return,h)}t.flags&=-3}e&4096&&(t.flags&=-4097)}function h_(t){if(t.subtreeFlags&1024)for(t=t.child;t!==null;){var e=t;h_(e),e.tag===5&&e.flags&1024&&e.stateNode.reset(),t=t.sibling}}function is(t,e){if(e.subtreeFlags&8772)for(e=e.child;e!==null;)a_(t,e.alternate,e),e=e.sibling}function Dr(t){for(t=t.child;t!==null;){var e=t;switch(e.tag){case 0:case 11:case 14:case 15:er(4,e,e.return),Dr(e);break;case 1:zi(e,e.return);var n=e.stateNode;typeof n.componentWillUnmount=="function"&&n_(e,e.return,n),Dr(e);break;case 27:tl(e.stateNode);case 26:case 5:zi(e,e.return),Dr(e);break;case 22:e.memoizedState===null&&Dr(e);break;case 30:Dr(e);break;default:Dr(e)}t=t.sibling}}function ss(t,e,n){for(n=n&&(e.subtreeFlags&8772)!==0,e=e.child;e!==null;){var i=e.alternate,s=t,r=e,a=r.flags;switch(r.tag){case 0:case 11:case 15:ss(s,r,n),El(4,r);break;case 1:if(ss(s,r,n),i=r,s=i.stateNode,typeof s.componentDidMount=="function")try{s.componentDidMount()}catch(c){ot(i,i.return,c)}if(i=r,s=i.updateQueue,s!==null){var o=i.stateNode;try{var l=s.shared.hiddenCallbacks;if(l!==null)for(s.shared.hiddenCallbacks=null,s=0;s<l.length;s++)cy(l[s],o)}catch(c){ot(i,i.return,c)}}n&&a&64&&t_(r),Jo(r,r.return);break;case 27:r_(r);case 26:case 5:ss(s,r,n),n&&i===null&&a&4&&i_(r),Jo(r,r.return);break;case 12:ss(s,r,n);break;case 31:ss(s,r,n),n&&a&4&&c_(s,r);break;case 13:ss(s,r,n),n&&a&4&&u_(s,r);break;case 22:r.memoizedState===null&&ss(s,r,n),Jo(r,r.return);break;case 30:break;default:ss(s,r,n)}e=e.sibling}}function mm(t,e){var n=null;t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),t=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(t=e.memoizedState.cachePool.pool),t!==n&&(t!=null&&t.refCount++,n!=null&&Al(n))}function gm(t,e){t=null,e.alternate!==null&&(t=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==t&&(e.refCount++,t!=null&&Al(t))}function _i(t,e,n,i){if(e.subtreeFlags&10256)for(e=e.child;e!==null;)d_(t,e,n,i),e=e.sibling}function d_(t,e,n,i){var s=e.flags;switch(e.tag){case 0:case 11:case 15:_i(t,e,n,i),s&2048&&El(9,e);break;case 1:_i(t,e,n,i);break;case 3:_i(t,e,n,i),s&2048&&(t=null,e.alternate!==null&&(t=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==t&&(e.refCount++,t!=null&&Al(t)));break;case 12:if(s&2048){_i(t,e,n,i),t=e.stateNode;try{var r=e.memoizedProps,a=r.id,o=r.onPostCommit;typeof o=="function"&&o(a,e.alternate===null?"mount":"update",t.passiveEffectDuration,-0)}catch(l){ot(e,e.return,l)}}else _i(t,e,n,i);break;case 31:_i(t,e,n,i);break;case 13:_i(t,e,n,i);break;case 23:break;case 22:r=e.stateNode,a=e.alternate,e.memoizedState!==null?r._visibility&2?_i(t,e,n,i):jo(t,e):r._visibility&2?_i(t,e,n,i):(r._visibility|=2,ga(t,e,n,i,(e.subtreeFlags&10256)!==0||!1)),s&2048&&mm(a,e);break;case 24:_i(t,e,n,i),s&2048&&gm(e.alternate,e);break;default:_i(t,e,n,i)}}function ga(t,e,n,i,s){for(s=s&&((e.subtreeFlags&10256)!==0||!1),e=e.child;e!==null;){var r=t,a=e,o=n,l=i,c=a.flags;switch(a.tag){case 0:case 11:case 15:ga(r,a,o,l,s),El(8,a);break;case 23:break;case 22:var h=a.stateNode;a.memoizedState!==null?h._visibility&2?ga(r,a,o,l,s):jo(r,a):(h._visibility|=2,ga(r,a,o,l,s)),s&&c&2048&&mm(a.alternate,a);break;case 24:ga(r,a,o,l,s),s&&c&2048&&gm(a.alternate,a);break;default:ga(r,a,o,l,s)}e=e.sibling}}function jo(t,e){if(e.subtreeFlags&10256)for(e=e.child;e!==null;){var n=t,i=e,s=i.flags;switch(i.tag){case 22:jo(n,i),s&2048&&mm(i.alternate,i);break;case 24:jo(n,i),s&2048&&gm(i.alternate,i);break;default:jo(n,i)}e=e.sibling}}var Vo=8192;function ma(t,e,n){if(t.subtreeFlags&Vo)for(t=t.child;t!==null;)p_(t,e,n),t=t.sibling}function p_(t,e,n){switch(t.tag){case 26:ma(t,e,n),t.flags&Vo&&t.memoizedState!==null&&kE(n,Si,t.memoizedState,t.memoizedProps);break;case 5:ma(t,e,n);break;case 3:case 4:var i=Si;Si=yu(t.stateNode.containerInfo),ma(t,e,n),Si=i;break;case 22:t.memoizedState===null&&(i=t.alternate,i!==null&&i.memoizedState!==null?(i=Vo,Vo=16777216,ma(t,e,n),Vo=i):ma(t,e,n));break;default:ma(t,e,n)}}function m_(t){var e=t.alternate;if(e!==null&&(t=e.child,t!==null)){e.child=null;do e=t.sibling,t.sibling=null,t=e;while(t!==null)}}function Lo(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];an=i,v_(i,t)}m_(t)}if(t.subtreeFlags&10256)for(t=t.child;t!==null;)g_(t),t=t.sibling}function g_(t){switch(t.tag){case 0:case 11:case 15:Lo(t),t.flags&2048&&er(9,t,t.return);break;case 3:Lo(t);break;case 12:Lo(t);break;case 22:var e=t.stateNode;t.memoizedState!==null&&e._visibility&2&&(t.return===null||t.return.tag!==13)?(e._visibility&=-3,Wc(t)):Lo(t);break;default:Lo(t)}}function Wc(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];an=i,v_(i,t)}m_(t)}for(t=t.child;t!==null;){switch(e=t,e.tag){case 0:case 11:case 15:er(8,e,e.return),Wc(e);break;case 22:n=e.stateNode,n._visibility&2&&(n._visibility&=-3,Wc(e));break;default:Wc(e)}t=t.sibling}}function v_(t,e){for(;an!==null;){var n=an;switch(n.tag){case 0:case 11:case 15:er(8,n,e);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var i=n.memoizedState.cachePool.pool;i!=null&&i.refCount++}break;case 24:Al(n.memoizedState.cache)}if(i=n.child,i!==null)i.return=n,an=i;else e:for(n=t;an!==null;){i=an;var s=i.sibling,r=i.return;if(o_(i),i===n){an=null;break e}if(s!==null){s.return=r,an=s;break e}an=r}}}var aE={getCacheForType:function(t){var e=mn(qt),n=e.data.get(t);return n===void 0&&(n=t(),e.data.set(t,n)),n},cacheSignal:function(){return mn(qt).controller.signal}},oE=typeof WeakMap=="function"?WeakMap:Map,nt=0,pt=null,Ve=null,Xe=0,at=0,Vn=null,Gs=!1,ja=!1,vm=!1,vs=0,Lt=0,tr=0,Lr=0,xm=0,Xn=0,ka=0,$o=null,Pn=null,vp=!1,Lu=0,x_=0,hu=1/0,du=null,qs=null,en=0,Qs=null,Wa=null,hs=0,xp=0,yp=null,y_=null,el=0,_p=null;function Zn(){return(nt&2)!==0&&Xe!==0?Xe&-Xe:Be.T!==null?_m():Cx()}function __(){if(Xn===0)if((Xe&536870912)===0||qe){var t=xc;xc<<=1,(xc&3932160)===0&&(xc=262144),Xn=t}else Xn=536870912;return t=Jn.current,t!==null&&(t.flags|=32),Xn}function Ln(t,e,n){(t===pt&&(at===2||at===9)||t.cancelPendingCommit!==null)&&(Xa(t,0),Hs(t,Xe,Xn,!1)),yl(t,n),((nt&2)===0||t!==pt)&&(t===pt&&((nt&2)===0&&(Lr|=n),Lt===4&&Hs(t,Xe,Xn,!1)),Vi(t))}function S_(t,e,n){if((nt&6)!==0)throw Error(J(327));var i=!n&&(e&127)===0&&(e&t.expiredLanes)===0||xl(t,e),s=i?uE(t,e):Dd(t,e,!0),r=i;do{if(s===0){ja&&!i&&Hs(t,e,0,!1);break}else{if(n=t.current.alternate,r&&!lE(n)){s=Dd(t,e,!1),r=!1;continue}if(s===2){if(r=e,t.errorRecoveryDisabledLanes&r)var a=0;else a=t.pendingLanes&-536870913,a=a!==0?a:a&536870912?536870912:0;if(a!==0){e=a;e:{var o=t;s=$o;var l=o.current.memoizedState.isDehydrated;if(l&&(Xa(o,a).flags|=256),a=Dd(o,a,!1),a!==2){if(vm&&!l){o.errorRecoveryDisabledLanes|=r,Lr|=r,s=4;break e}r=Pn,Pn=s,r!==null&&(Pn===null?Pn=r:Pn.push.apply(Pn,r))}s=a}if(r=!1,s!==2)continue}}if(s===1){Xa(t,0),Hs(t,e,0,!0);break}e:{switch(i=t,r=s,r){case 0:case 1:throw Error(J(345));case 4:if((e&4194048)!==e)break;case 6:Hs(i,e,Xn,!Gs);break e;case 2:Pn=null;break;case 3:case 5:break;default:throw Error(J(329))}if((e&62914560)===e&&(s=Lu+300-Yn(),10<s)){if(Hs(i,e,Xn,!Gs),Eu(i,0,!0)!==0)break e;hs=e,i.timeoutHandle=H_(z0.bind(null,i,n,Pn,du,vp,e,Xn,Lr,ka,Gs,r,"Throttled",-0,0),s);break e}z0(i,n,Pn,du,vp,e,Xn,Lr,ka,Gs,r,null,-0,0)}}break}while(!0);Vi(t)}function z0(t,e,n,i,s,r,a,o,l,c,h,p,u,d){if(t.timeoutHandle=-1,p=e.subtreeFlags,p&8192||(p&16785408)===16785408){p={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:ls},p_(e,r,p);var v=(r&62914560)===r?Lu-Yn():(r&4194048)===r?x_-Yn():0;if(v=WE(p,v),v!==null){hs=r,t.cancelPendingCommit=v(H0.bind(null,t,e,r,n,i,s,a,o,l,h,p,null,u,d)),Hs(t,r,a,!c);return}}H0(t,e,r,n,i,s,a,o,l)}function lE(t){for(var e=t;;){var n=e.tag;if((n===0||n===11||n===15)&&e.flags&16384&&(n=e.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var i=0;i<n.length;i++){var s=n[i],r=s.getSnapshot;s=s.value;try{if(!Kn(r(),s))return!1}catch{return!1}}if(n=e.child,e.subtreeFlags&16384&&n!==null)n.return=e,e=n;else{if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return!0;e=e.return}e.sibling.return=e.return,e=e.sibling}}return!0}function Hs(t,e,n,i){e&=~xm,e&=~Lr,t.suspendedLanes|=e,t.pingedLanes&=~e,i&&(t.warmLanes|=e),i=t.expirationTimes;for(var s=e;0<s;){var r=31-Qn(s),a=1<<r;i[r]=-1,s&=~a}n!==0&&Tx(t,n,e)}function Nu(){return(nt&6)===0?(Tl(0,!1),!1):!0}function ym(){if(Ve!==null){if(at===0)var t=Ve.return;else t=Ve,cs=Xr=null,sm(t),Pa=null,ll=0,t=Ve;for(;t!==null;)e_(t.alternate,t),t=t.return;Ve=null}}function Xa(t,e){var n=t.timeoutHandle;n!==-1&&(t.timeoutHandle=-1,bE(n)),n=t.cancelPendingCommit,n!==null&&(t.cancelPendingCommit=null,n()),hs=0,ym(),pt=t,Ve=n=us(t.current,null),Xe=e,at=0,Vn=null,Gs=!1,ja=xl(t,e),vm=!1,ka=Xn=xm=Lr=tr=Lt=0,Pn=$o=null,vp=!1,(e&8)!==0&&(e|=e&32);var i=t.entangledLanes;if(i!==0)for(t=t.entanglements,i&=e;0<i;){var s=31-Qn(i),r=1<<s;e|=t[s],i&=~r}return vs=e,Cu(),n}function A_(t,e){Fe=null,Be.H=ul,e===Ja||e===Du?(e=v0(),at=3):e===Jp?(e=v0(),at=4):at=e===dm?8:e!==null&&typeof e=="object"&&typeof e.then=="function"?6:1,Vn=e,Ve===null&&(Lt=1,cu(t,ui(e,t.current)))}function M_(){var t=Jn.current;return t===null?!0:(Xe&4194048)===Xe?hi===null:(Xe&62914560)===Xe||(Xe&536870912)!==0?t===hi:!1}function E_(){var t=Be.H;return Be.H=ul,t===null?ul:t}function T_(){var t=Be.A;return Be.A=aE,t}function pu(){Lt=4,Gs||(Xe&4194048)!==Xe&&Jn.current!==null||(ja=!0),(tr&134217727)===0&&(Lr&134217727)===0||pt===null||Hs(pt,Xe,Xn,!1)}function Dd(t,e,n){var i=nt;nt|=2;var s=E_(),r=T_();(pt!==t||Xe!==e)&&(du=null,Xa(t,e)),e=!1;var a=Lt;e:do try{if(at!==0&&Ve!==null){var o=Ve,l=Vn;switch(at){case 8:ym(),a=6;break e;case 3:case 2:case 9:case 6:Jn.current===null&&(e=!0);var c=at;if(at=0,Vn=null,Ra(t,o,l,c),n&&ja){a=0;break e}break;default:c=at,at=0,Vn=null,Ra(t,o,l,c)}}cE(),a=Lt;break}catch(h){A_(t,h)}while(!0);return e&&t.shellSuspendCounter++,cs=Xr=null,nt=i,Be.H=s,Be.A=r,Ve===null&&(pt=null,Xe=0,Cu()),a}function cE(){for(;Ve!==null;)b_(Ve)}function uE(t,e){var n=nt;nt|=2;var i=E_(),s=T_();pt!==t||Xe!==e?(du=null,hu=Yn()+500,Xa(t,e)):ja=xl(t,e);e:do try{if(at!==0&&Ve!==null){e=Ve;var r=Vn;t:switch(at){case 1:at=0,Vn=null,Ra(t,e,r,1);break;case 2:case 9:if(g0(r)){at=0,Vn=null,G0(e);break}e=function(){at!==2&&at!==9||pt!==t||(at=7),Vi(t)},r.then(e,e);break e;case 3:at=7;break e;case 4:at=5;break e;case 7:g0(r)?(at=0,Vn=null,G0(e)):(at=0,Vn=null,Ra(t,e,r,7));break;case 5:var a=null;switch(Ve.tag){case 26:a=Ve.memoizedState;case 5:case 27:var o=Ve;if(a?Y_(a):o.stateNode.complete){at=0,Vn=null;var l=o.sibling;if(l!==null)Ve=l;else{var c=o.return;c!==null?(Ve=c,Ou(c)):Ve=null}break t}}at=0,Vn=null,Ra(t,e,r,5);break;case 6:at=0,Vn=null,Ra(t,e,r,6);break;case 8:ym(),Lt=6;break e;default:throw Error(J(462))}}fE();break}catch(h){A_(t,h)}while(!0);return cs=Xr=null,Be.H=i,Be.A=s,nt=n,Ve!==null?0:(pt=null,Xe=0,Cu(),Lt)}function fE(){for(;Ve!==null&&!P1();)b_(Ve)}function b_(t){var e=$y(t.alternate,t,vs);t.memoizedProps=t.pendingProps,e===null?Ou(t):Ve=e}function G0(t){var e=t,n=e.alternate;switch(e.tag){case 15:case 0:e=I0(n,e,e.pendingProps,e.type,void 0,Xe);break;case 11:e=I0(n,e,e.pendingProps,e.type.render,e.ref,Xe);break;case 5:sm(e);default:e_(n,e),e=Ve=ey(e,vs),e=$y(n,e,vs)}t.memoizedProps=t.pendingProps,e===null?Ou(t):Ve=e}function Ra(t,e,n,i){cs=Xr=null,sm(e),Pa=null,ll=0;var s=e.return;try{if($M(t,s,e,n,Xe)){Lt=1,cu(t,ui(n,t.current)),Ve=null;return}}catch(r){if(s!==null)throw Ve=s,r;Lt=1,cu(t,ui(n,t.current)),Ve=null;return}e.flags&32768?(qe||i===1?t=!0:ja||(Xe&536870912)!==0?t=!1:(Gs=t=!0,(i===2||i===9||i===3||i===6)&&(i=Jn.current,i!==null&&i.tag===13&&(i.flags|=16384))),w_(e,t)):Ou(e)}function Ou(t){var e=t;do{if((e.flags&32768)!==0){w_(e,Gs);return}t=e.return;var n=nE(e.alternate,e,vs);if(n!==null){Ve=n;return}if(e=e.sibling,e!==null){Ve=e;return}Ve=e=t}while(e!==null);Lt===0&&(Lt=5)}function w_(t,e){do{var n=iE(t.alternate,t);if(n!==null){n.flags&=32767,Ve=n;return}if(n=t.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!e&&(t=t.sibling,t!==null)){Ve=t;return}Ve=t=n}while(t!==null);Lt=6,Ve=null}function H0(t,e,n,i,s,r,a,o,l){t.cancelPendingCommit=null;do Fu();while(en!==0);if((nt&6)!==0)throw Error(J(327));if(e!==null){if(e===t.current)throw Error(J(177));if(r=e.lanes|e.childLanes,r|=Wp,W1(t,n,r,a,o,l),t===pt&&(Ve=pt=null,Xe=0),Wa=e,Qs=t,hs=n,xp=r,yp=s,y_=i,(e.subtreeFlags&10256)!==0||(e.flags&10256)!==0?(t.callbackNode=null,t.callbackPriority=0,mE(jc,function(){return B_(),null})):(t.callbackNode=null,t.callbackPriority=0),i=(e.flags&13878)!==0,(e.subtreeFlags&13878)!==0||i){i=Be.T,Be.T=null,s=it.p,it.p=2,a=nt,nt|=4;try{sE(t,e,n)}finally{nt=a,it.p=s,Be.T=i}}en=1,C_(),R_(),D_()}}function C_(){if(en===1){en=0;var t=Qs,e=Wa,n=(e.flags&13878)!==0;if((e.subtreeFlags&13878)!==0||n){n=Be.T,Be.T=null;var i=it.p;it.p=2;var s=nt;nt|=4;try{f_(e,t);var r=Ep,a=Yx(t.containerInfo),o=r.focusedElem,l=r.selectionRange;if(a!==o&&o&&o.ownerDocument&&Xx(o.ownerDocument.documentElement,o)){if(l!==null&&kp(o)){var c=l.start,h=l.end;if(h===void 0&&(h=c),"selectionStart"in o)o.selectionStart=c,o.selectionEnd=Math.min(h,o.value.length);else{var p=o.ownerDocument||document,u=p&&p.defaultView||window;if(u.getSelection){var d=u.getSelection(),v=o.textContent.length,M=Math.min(l.start,v),m=l.end===void 0?M:Math.min(l.end,v);!d.extend&&M>m&&(a=m,m=M,M=a);var f=c0(o,M),g=c0(o,m);if(f&&g&&(d.rangeCount!==1||d.anchorNode!==f.node||d.anchorOffset!==f.offset||d.focusNode!==g.node||d.focusOffset!==g.offset)){var S=p.createRange();S.setStart(f.node,f.offset),d.removeAllRanges(),M>m?(d.addRange(S),d.extend(g.node,g.offset)):(S.setEnd(g.node,g.offset),d.addRange(S))}}}}for(p=[],d=o;d=d.parentNode;)d.nodeType===1&&p.push({element:d,left:d.scrollLeft,top:d.scrollTop});for(typeof o.focus=="function"&&o.focus(),o=0;o<p.length;o++){var _=p[o];_.element.scrollLeft=_.left,_.element.scrollTop=_.top}}Au=!!Mp,Ep=Mp=null}finally{nt=s,it.p=i,Be.T=n}}t.current=e,en=2}}function R_(){if(en===2){en=0;var t=Qs,e=Wa,n=(e.flags&8772)!==0;if((e.subtreeFlags&8772)!==0||n){n=Be.T,Be.T=null;var i=it.p;it.p=2;var s=nt;nt|=4;try{a_(t,e.alternate,e)}finally{nt=s,it.p=i,Be.T=n}}en=3}}function D_(){if(en===4||en===3){en=0,L1();var t=Qs,e=Wa,n=hs,i=y_;(e.subtreeFlags&10256)!==0||(e.flags&10256)!==0?en=5:(en=0,Wa=Qs=null,U_(t,t.pendingLanes));var s=t.pendingLanes;if(s===0&&(qs=null),Np(n),e=e.stateNode,qn&&typeof qn.onCommitFiberRoot=="function")try{qn.onCommitFiberRoot(vl,e,void 0,(e.current.flags&128)===128)}catch{}if(i!==null){e=Be.T,s=it.p,it.p=2,Be.T=null;try{for(var r=t.onRecoverableError,a=0;a<i.length;a++){var o=i[a];r(o.value,{componentStack:o.stack})}}finally{Be.T=e,it.p=s}}(hs&3)!==0&&Fu(),Vi(t),s=t.pendingLanes,(n&261930)!==0&&(s&42)!==0?t===_p?el++:(el=0,_p=t):el=0,Tl(0,!1)}}function U_(t,e){(t.pooledCacheLanes&=e)===0&&(e=t.pooledCache,e!=null&&(t.pooledCache=null,Al(e)))}function Fu(){return C_(),R_(),D_(),B_()}function B_(){if(en!==5)return!1;var t=Qs,e=xp;xp=0;var n=Np(hs),i=Be.T,s=it.p;try{it.p=32>n?32:n,Be.T=null,n=yp,yp=null;var r=Qs,a=hs;if(en=0,Wa=Qs=null,hs=0,(nt&6)!==0)throw Error(J(331));var o=nt;if(nt|=4,g_(r.current),d_(r,r.current,a,n),nt=o,Tl(0,!1),qn&&typeof qn.onPostCommitFiberRoot=="function")try{qn.onPostCommitFiberRoot(vl,r)}catch{}return!0}finally{it.p=s,Be.T=i,U_(t,e)}}function V0(t,e,n){e=ui(n,e),e=dp(t.stateNode,e,2),t=Ys(t,e,2),t!==null&&(yl(t,2),Vi(t))}function ot(t,e,n){if(t.tag===3)V0(t,t,n);else for(;e!==null;){if(e.tag===3){V0(e,t,n);break}else if(e.tag===1){var i=e.stateNode;if(typeof e.type.getDerivedStateFromError=="function"||typeof i.componentDidCatch=="function"&&(qs===null||!qs.has(i))){t=ui(n,t),n=qy(2),i=Ys(e,n,2),i!==null&&(Qy(n,i,e,t),yl(i,2),Vi(i));break}}e=e.return}}function Ud(t,e,n){var i=t.pingCache;if(i===null){i=t.pingCache=new oE;var s=new Set;i.set(e,s)}else s=i.get(e),s===void 0&&(s=new Set,i.set(e,s));s.has(n)||(vm=!0,s.add(n),t=hE.bind(null,t,e,n),e.then(t,t))}function hE(t,e,n){var i=t.pingCache;i!==null&&i.delete(e),t.pingedLanes|=t.suspendedLanes&n,t.warmLanes&=~n,pt===t&&(Xe&n)===n&&(Lt===4||Lt===3&&(Xe&62914560)===Xe&&300>Yn()-Lu?(nt&2)===0&&Xa(t,0):xm|=n,ka===Xe&&(ka=0)),Vi(t)}function I_(t,e){e===0&&(e=Ex()),t=Wr(t,e),t!==null&&(yl(t,e),Vi(t))}function dE(t){var e=t.memoizedState,n=0;e!==null&&(n=e.retryLane),I_(t,n)}function pE(t,e){var n=0;switch(t.tag){case 31:case 13:var i=t.stateNode,s=t.memoizedState;s!==null&&(n=s.retryLane);break;case 19:i=t.stateNode;break;case 22:i=t.stateNode._retryCache;break;default:throw Error(J(314))}i!==null&&i.delete(e),I_(t,n)}function mE(t,e){return Pp(t,e)}var mu=null,va=null,Sp=!1,gu=!1,Bd=!1,Vs=0;function Vi(t){t!==va&&t.next===null&&(va===null?mu=va=t:va=va.next=t),gu=!0,Sp||(Sp=!0,vE())}function Tl(t,e){if(!Bd&&gu){Bd=!0;do for(var n=!1,i=mu;i!==null;){if(!e)if(t!==0){var s=i.pendingLanes;if(s===0)var r=0;else{var a=i.suspendedLanes,o=i.pingedLanes;r=(1<<31-Qn(42|t)+1)-1,r&=s&~(a&~o),r=r&201326741?r&201326741|1:r?r|2:0}r!==0&&(n=!0,k0(i,r))}else r=Xe,r=Eu(i,i===pt?r:0,i.cancelPendingCommit!==null||i.timeoutHandle!==-1),(r&3)===0||xl(i,r)||(n=!0,k0(i,r));i=i.next}while(n);Bd=!1}}function gE(){P_()}function P_(){gu=Sp=!1;var t=0;Vs!==0&&TE()&&(t=Vs);for(var e=Yn(),n=null,i=mu;i!==null;){var s=i.next,r=L_(i,e);r===0?(i.next=null,n===null?mu=s:n.next=s,s===null&&(va=n)):(n=i,(t!==0||(r&3)!==0)&&(gu=!0)),i=s}en!==0&&en!==5||Tl(t,!1),Vs!==0&&(Vs=0)}function L_(t,e){for(var n=t.suspendedLanes,i=t.pingedLanes,s=t.expirationTimes,r=t.pendingLanes&-62914561;0<r;){var a=31-Qn(r),o=1<<a,l=s[a];l===-1?((o&n)===0||(o&i)!==0)&&(s[a]=k1(o,e)):l<=e&&(t.expiredLanes|=o),r&=~o}if(e=pt,n=Xe,n=Eu(t,t===e?n:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i=t.callbackNode,n===0||t===e&&(at===2||at===9)||t.cancelPendingCommit!==null)return i!==null&&i!==null&&od(i),t.callbackNode=null,t.callbackPriority=0;if((n&3)===0||xl(t,n)){if(e=n&-n,e===t.callbackPriority)return e;switch(i!==null&&od(i),Np(n)){case 2:case 8:n=Ax;break;case 32:n=jc;break;case 268435456:n=Mx;break;default:n=jc}return i=N_.bind(null,t),n=Pp(n,i),t.callbackPriority=e,t.callbackNode=n,e}return i!==null&&i!==null&&od(i),t.callbackPriority=2,t.callbackNode=null,2}function N_(t,e){if(en!==0&&en!==5)return t.callbackNode=null,t.callbackPriority=0,null;var n=t.callbackNode;if(Fu()&&t.callbackNode!==n)return null;var i=Xe;return i=Eu(t,t===pt?i:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i===0?null:(S_(t,i,e),L_(t,Yn()),t.callbackNode!=null&&t.callbackNode===n?N_.bind(null,t):null)}function k0(t,e){if(Fu())return null;S_(t,e,!0)}function vE(){wE(function(){(nt&6)!==0?Pp(Sx,gE):P_()})}function _m(){if(Vs===0){var t=Ga;t===0&&(t=vc,vc<<=1,(vc&261888)===0&&(vc=256)),Vs=t}return Vs}function W0(t){return t==null||typeof t=="symbol"||typeof t=="boolean"?null:typeof t=="function"?t:Lc(""+t)}function X0(t,e){var n=e.ownerDocument.createElement("input");return n.name=e.name,n.value=e.value,t.id&&n.setAttribute("form",t.id),e.parentNode.insertBefore(n,e),t=new FormData(t),n.parentNode.removeChild(n),t}function xE(t,e,n,i,s){if(e==="submit"&&n&&n.stateNode===s){var r=W0((s[Nn]||null).action),a=i.submitter;a&&(e=(e=a[Nn]||null)?W0(e.formAction):a.getAttribute("formAction"),e!==null&&(r=e,a=null));var o=new Tu("action","action",null,i,s);t.push({event:o,listeners:[{instance:null,listener:function(){if(i.defaultPrevented){if(Vs!==0){var l=a?X0(s,a):new FormData(s);fp(n,{pending:!0,data:l,method:s.method,action:r},null,l)}}else typeof r=="function"&&(o.preventDefault(),l=a?X0(s,a):new FormData(s),fp(n,{pending:!0,data:l,method:s.method,action:r},r,l))},currentTarget:s}]})}}for(Rc=0;Rc<$d.length;Rc++)Dc=$d[Rc],Y0=Dc.toLowerCase(),q0=Dc[0].toUpperCase()+Dc.slice(1),Ai(Y0,"on"+q0);var Dc,Y0,q0,Rc;Ai(Qx,"onAnimationEnd");Ai(Zx,"onAnimationIteration");Ai(Kx,"onAnimationStart");Ai("dblclick","onDoubleClick");Ai("focusin","onFocus");Ai("focusout","onBlur");Ai(NM,"onTransitionRun");Ai(OM,"onTransitionStart");Ai(FM,"onTransitionCancel");Ai(Jx,"onTransitionEnd");Fa("onMouseEnter",["mouseout","mouseover"]);Fa("onMouseLeave",["mouseout","mouseover"]);Fa("onPointerEnter",["pointerout","pointerover"]);Fa("onPointerLeave",["pointerout","pointerover"]);Hr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Hr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Hr("onBeforeInput",["compositionend","keypress","textInput","paste"]);Hr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Hr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Hr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var fl="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),yE=new Set("beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(fl));function O_(t,e){e=(e&4)!==0;for(var n=0;n<t.length;n++){var i=t[n],s=i.event;i=i.listeners;e:{var r=void 0;if(e)for(var a=i.length-1;0<=a;a--){var o=i[a],l=o.instance,c=o.currentTarget;if(o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){eu(h)}s.currentTarget=null,r=l}else for(a=0;a<i.length;a++){if(o=i[a],l=o.instance,c=o.currentTarget,o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){eu(h)}s.currentTarget=null,r=l}}}}function He(t,e){var n=e[Xd];n===void 0&&(n=e[Xd]=new Set);var i=t+"__bubble";n.has(i)||(F_(e,t,2,!1),n.add(i))}function Id(t,e,n){var i=0;e&&(i|=4),F_(n,t,i,e)}var Uc="_reactListening"+Math.random().toString(36).slice(2);function Sm(t){if(!t[Uc]){t[Uc]=!0,Rx.forEach(function(n){n!=="selectionchange"&&(yE.has(n)||Id(n,!1,t),Id(n,!0,t))});var e=t.nodeType===9?t:t.ownerDocument;e===null||e[Uc]||(e[Uc]=!0,Id("selectionchange",!1,e))}}function F_(t,e,n,i){switch(J_(e)){case 2:var s=qE;break;case 8:s=QE;break;default:s=Tm}n=s.bind(null,e,n,t),s=void 0,!Kd||e!=="touchstart"&&e!=="touchmove"&&e!=="wheel"||(s=!0),i?s!==void 0?t.addEventListener(e,n,{capture:!0,passive:s}):t.addEventListener(e,n,!0):s!==void 0?t.addEventListener(e,n,{passive:s}):t.addEventListener(e,n,!1)}function Pd(t,e,n,i,s){var r=i;if((e&1)===0&&(e&2)===0&&i!==null)e:for(;;){if(i===null)return;var a=i.tag;if(a===3||a===4){var o=i.stateNode.containerInfo;if(o===s)break;if(a===4)for(a=i.return;a!==null;){var l=a.tag;if((l===3||l===4)&&a.stateNode.containerInfo===s)return;a=a.return}for(;o!==null;){if(a=_a(o),a===null)return;if(l=a.tag,l===5||l===6||l===26||l===27){i=r=a;continue e}o=o.parentNode}}i=i.return}Ox(function(){var c=r,h=zp(n),p=[];e:{var u=jx.get(t);if(u!==void 0){var d=Tu,v=t;switch(t){case"keypress":if(Oc(n)===0)break e;case"keydown":case"keyup":d=pM;break;case"focusin":v="focus",d=hd;break;case"focusout":v="blur",d=hd;break;case"beforeblur":case"afterblur":d=hd;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":d=e0;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":d=nM;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":d=vM;break;case Qx:case Zx:case Kx:d=rM;break;case Jx:d=yM;break;case"scroll":case"scrollend":d=eM;break;case"wheel":d=SM;break;case"copy":case"cut":case"paste":d=oM;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":d=n0;break;case"toggle":case"beforetoggle":d=MM}var M=(e&4)!==0,m=!M&&(t==="scroll"||t==="scrollend"),f=M?u!==null?u+"Capture":null:u;M=[];for(var g=c,S;g!==null;){var _=g;if(S=_.stateNode,_=_.tag,_!==5&&_!==26&&_!==27||S===null||f===null||(_=il(g,f),_!=null&&M.push(hl(g,_,S))),m)break;g=g.return}0<M.length&&(u=new d(u,v,null,n,h),p.push({event:u,listeners:M}))}}if((e&7)===0){e:{if(u=t==="mouseover"||t==="pointerover",d=t==="mouseout"||t==="pointerout",u&&n!==Zd&&(v=n.relatedTarget||n.fromElement)&&(_a(v)||v[Qa]))break e;if((d||u)&&(u=h.window===h?h:(u=h.ownerDocument)?u.defaultView||u.parentWindow:window,d?(v=n.relatedTarget||n.toElement,d=c,v=v?_a(v):null,v!==null&&(m=gl(v),M=v.tag,v!==m||M!==5&&M!==27&&M!==6)&&(v=null)):(d=null,v=c),d!==v)){if(M=e0,_="onMouseLeave",f="onMouseEnter",g="mouse",(t==="pointerout"||t==="pointerover")&&(M=n0,_="onPointerLeave",f="onPointerEnter",g="pointer"),m=d==null?u:Go(d),S=v==null?u:Go(v),u=new M(_,g+"leave",d,n,h),u.target=m,u.relatedTarget=S,_=null,_a(h)===c&&(M=new M(f,g+"enter",v,n,h),M.target=S,M.relatedTarget=m,_=M),m=_,d&&v)t:{for(M=_E,f=d,g=v,S=0,_=f;_;_=M(_))S++;_=0;for(var T=g;T;T=M(T))_++;for(;0<S-_;)f=M(f),S--;for(;0<_-S;)g=M(g),_--;for(;S--;){if(f===g||g!==null&&f===g.alternate){M=f;break t}f=M(f),g=M(g)}M=null}else M=null;d!==null&&Q0(p,u,d,M,!1),v!==null&&m!==null&&Q0(p,m,v,M,!0)}}e:{if(u=c?Go(c):window,d=u.nodeName&&u.nodeName.toLowerCase(),d==="select"||d==="input"&&u.type==="file")var b=a0;else if(r0(u))if(kx)b=IM;else{b=UM;var w=DM}else d=u.nodeName,!d||d.toLowerCase()!=="input"||u.type!=="checkbox"&&u.type!=="radio"?c&&Fp(c.elementType)&&(b=a0):b=BM;if(b&&(b=b(t,c))){Vx(p,b,n,h);break e}w&&w(t,u,c),t==="focusout"&&c&&u.type==="number"&&c.memoizedProps.value!=null&&Qd(u,"number",u.value)}switch(w=c?Go(c):window,t){case"focusin":(r0(w)||w.contentEditable==="true")&&(Ma=w,Jd=c,Xo=null);break;case"focusout":Xo=Jd=Ma=null;break;case"mousedown":jd=!0;break;case"contextmenu":case"mouseup":case"dragend":jd=!1,u0(p,n,h);break;case"selectionchange":if(LM)break;case"keydown":case"keyup":u0(p,n,h)}var x;if(Vp)e:{switch(t){case"compositionstart":var E="onCompositionStart";break e;case"compositionend":E="onCompositionEnd";break e;case"compositionupdate":E="onCompositionUpdate";break e}E=void 0}else Aa?Gx(t,n)&&(E="onCompositionEnd"):t==="keydown"&&n.keyCode===229&&(E="onCompositionStart");E&&(zx&&n.locale!=="ko"&&(Aa||E!=="onCompositionStart"?E==="onCompositionEnd"&&Aa&&(x=Fx()):(zs=h,Gp="value"in zs?zs.value:zs.textContent,Aa=!0)),w=vu(c,E),0<w.length&&(E=new t0(E,t,null,n,h),p.push({event:E,listeners:w}),x?E.data=x:(x=Hx(n),x!==null&&(E.data=x)))),(x=TM?bM(t,n):wM(t,n))&&(E=vu(c,"onBeforeInput"),0<E.length&&(w=new t0("onBeforeInput","beforeinput",null,n,h),p.push({event:w,listeners:E}),w.data=x)),xE(p,t,c,n,h)}O_(p,e)})}function hl(t,e,n){return{instance:t,listener:e,currentTarget:n}}function vu(t,e){for(var n=e+"Capture",i=[];t!==null;){var s=t,r=s.stateNode;if(s=s.tag,s!==5&&s!==26&&s!==27||r===null||(s=il(t,n),s!=null&&i.unshift(hl(t,s,r)),s=il(t,e),s!=null&&i.push(hl(t,s,r))),t.tag===3)return i;t=t.return}return[]}function _E(t){if(t===null)return null;do t=t.return;while(t&&t.tag!==5&&t.tag!==27);return t||null}function Q0(t,e,n,i,s){for(var r=e._reactName,a=[];n!==null&&n!==i;){var o=n,l=o.alternate,c=o.stateNode;if(o=o.tag,l!==null&&l===i)break;o!==5&&o!==26&&o!==27||c===null||(l=c,s?(c=il(n,r),c!=null&&a.unshift(hl(n,c,l))):s||(c=il(n,r),c!=null&&a.push(hl(n,c,l)))),n=n.return}a.length!==0&&t.push({event:e,listeners:a})}var SE=/\r\n?/g,AE=/\u0000|\uFFFD/g;function Z0(t){return(typeof t=="string"?t:""+t).replace(SE,`
`).replace(AE,"")}function z_(t,e){return e=Z0(e),Z0(t)===e}function ct(t,e,n,i,s,r){switch(n){case"children":typeof i=="string"?e==="body"||e==="textarea"&&i===""||za(t,i):(typeof i=="number"||typeof i=="bigint")&&e!=="body"&&za(t,""+i);break;case"className":_c(t,"class",i);break;case"tabIndex":_c(t,"tabindex",i);break;case"dir":case"role":case"viewBox":case"width":case"height":_c(t,n,i);break;case"style":Nx(t,i,r);break;case"data":if(e!=="object"){_c(t,"data",i);break}case"src":case"href":if(i===""&&(e!=="a"||n!=="href")){t.removeAttribute(n);break}if(i==null||typeof i=="function"||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=Lc(""+i),t.setAttribute(n,i);break;case"action":case"formAction":if(typeof i=="function"){t.setAttribute(n,"javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')");break}else typeof r=="function"&&(n==="formAction"?(e!=="input"&&ct(t,e,"name",s.name,s,null),ct(t,e,"formEncType",s.formEncType,s,null),ct(t,e,"formMethod",s.formMethod,s,null),ct(t,e,"formTarget",s.formTarget,s,null)):(ct(t,e,"encType",s.encType,s,null),ct(t,e,"method",s.method,s,null),ct(t,e,"target",s.target,s,null)));if(i==null||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=Lc(""+i),t.setAttribute(n,i);break;case"onClick":i!=null&&(t.onclick=ls);break;case"onScroll":i!=null&&He("scroll",t);break;case"onScrollEnd":i!=null&&He("scrollend",t);break;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));t.innerHTML=n}}break;case"multiple":t.multiple=i&&typeof i!="function"&&typeof i!="symbol";break;case"muted":t.muted=i&&typeof i!="function"&&typeof i!="symbol";break;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"defaultValue":case"defaultChecked":case"innerHTML":case"ref":break;case"autoFocus":break;case"xlinkHref":if(i==null||typeof i=="function"||typeof i=="boolean"||typeof i=="symbol"){t.removeAttribute("xlink:href");break}n=Lc(""+i),t.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",n);break;case"contentEditable":case"spellCheck":case"draggable":case"value":case"autoReverse":case"externalResourcesRequired":case"focusable":case"preserveAlpha":i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,""+i):t.removeAttribute(n);break;case"inert":case"allowFullScreen":case"async":case"autoPlay":case"controls":case"default":case"defer":case"disabled":case"disablePictureInPicture":case"disableRemotePlayback":case"formNoValidate":case"hidden":case"loop":case"noModule":case"noValidate":case"open":case"playsInline":case"readOnly":case"required":case"reversed":case"scoped":case"seamless":case"itemScope":i&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,""):t.removeAttribute(n);break;case"capture":case"download":i===!0?t.setAttribute(n,""):i!==!1&&i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,i):t.removeAttribute(n);break;case"cols":case"rows":case"size":case"span":i!=null&&typeof i!="function"&&typeof i!="symbol"&&!isNaN(i)&&1<=i?t.setAttribute(n,i):t.removeAttribute(n);break;case"rowSpan":case"start":i==null||typeof i=="function"||typeof i=="symbol"||isNaN(i)?t.removeAttribute(n):t.setAttribute(n,i);break;case"popover":He("beforetoggle",t),He("toggle",t),Pc(t,"popover",i);break;case"xlinkActuate":es(t,"http://www.w3.org/1999/xlink","xlink:actuate",i);break;case"xlinkArcrole":es(t,"http://www.w3.org/1999/xlink","xlink:arcrole",i);break;case"xlinkRole":es(t,"http://www.w3.org/1999/xlink","xlink:role",i);break;case"xlinkShow":es(t,"http://www.w3.org/1999/xlink","xlink:show",i);break;case"xlinkTitle":es(t,"http://www.w3.org/1999/xlink","xlink:title",i);break;case"xlinkType":es(t,"http://www.w3.org/1999/xlink","xlink:type",i);break;case"xmlBase":es(t,"http://www.w3.org/XML/1998/namespace","xml:base",i);break;case"xmlLang":es(t,"http://www.w3.org/XML/1998/namespace","xml:lang",i);break;case"xmlSpace":es(t,"http://www.w3.org/XML/1998/namespace","xml:space",i);break;case"is":Pc(t,"is",i);break;case"innerText":case"textContent":break;default:(!(2<n.length)||n[0]!=="o"&&n[0]!=="O"||n[1]!=="n"&&n[1]!=="N")&&(n=j1.get(n)||n,Pc(t,n,i))}}function Ap(t,e,n,i,s,r){switch(n){case"style":Nx(t,i,r);break;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));t.innerHTML=n}}break;case"children":typeof i=="string"?za(t,i):(typeof i=="number"||typeof i=="bigint")&&za(t,""+i);break;case"onScroll":i!=null&&He("scroll",t);break;case"onScrollEnd":i!=null&&He("scrollend",t);break;case"onClick":i!=null&&(t.onclick=ls);break;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"innerHTML":case"ref":break;case"innerText":case"textContent":break;default:if(!Dx.hasOwnProperty(n))e:{if(n[0]==="o"&&n[1]==="n"&&(s=n.endsWith("Capture"),e=n.slice(2,s?n.length-7:void 0),r=t[Nn]||null,r=r!=null?r[n]:null,typeof r=="function"&&t.removeEventListener(e,r,s),typeof i=="function")){typeof r!="function"&&r!==null&&(n in t?t[n]=null:t.hasAttribute(n)&&t.removeAttribute(n)),t.addEventListener(e,i,s);break e}n in t?t[n]=i:i===!0?t.setAttribute(n,""):Pc(t,n,i)}}}function gn(t,e,n){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"img":He("error",t),He("load",t);var i=!1,s=!1,r;for(r in n)if(n.hasOwnProperty(r)){var a=n[r];if(a!=null)switch(r){case"src":i=!0;break;case"srcSet":s=!0;break;case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:ct(t,e,r,a,n,null)}}s&&ct(t,e,"srcSet",n.srcSet,n,null),i&&ct(t,e,"src",n.src,n,null);return;case"input":He("invalid",t);var o=r=a=s=null,l=null,c=null;for(i in n)if(n.hasOwnProperty(i)){var h=n[i];if(h!=null)switch(i){case"name":s=h;break;case"type":a=h;break;case"checked":l=h;break;case"defaultChecked":c=h;break;case"value":r=h;break;case"defaultValue":o=h;break;case"children":case"dangerouslySetInnerHTML":if(h!=null)throw Error(J(137,e));break;default:ct(t,e,i,h,n,null)}}Ix(t,r,o,l,c,a,s,!1);return;case"select":He("invalid",t),i=a=r=null;for(s in n)if(n.hasOwnProperty(s)&&(o=n[s],o!=null))switch(s){case"value":r=o;break;case"defaultValue":a=o;break;case"multiple":i=o;default:ct(t,e,s,o,n,null)}e=r,n=a,t.multiple=!!i,e!=null?Ua(t,!!i,e,!1):n!=null&&Ua(t,!!i,n,!0);return;case"textarea":He("invalid",t),r=s=i=null;for(a in n)if(n.hasOwnProperty(a)&&(o=n[a],o!=null))switch(a){case"value":i=o;break;case"defaultValue":s=o;break;case"children":r=o;break;case"dangerouslySetInnerHTML":if(o!=null)throw Error(J(91));break;default:ct(t,e,a,o,n,null)}Lx(t,i,s,r);return;case"option":for(l in n)n.hasOwnProperty(l)&&(i=n[l],i!=null)&&(l==="selected"?t.selected=i&&typeof i!="function"&&typeof i!="symbol":ct(t,e,l,i,n,null));return;case"dialog":He("beforetoggle",t),He("toggle",t),He("cancel",t),He("close",t);break;case"iframe":case"object":He("load",t);break;case"video":case"audio":for(i=0;i<fl.length;i++)He(fl[i],t);break;case"image":He("error",t),He("load",t);break;case"details":He("toggle",t);break;case"embed":case"source":case"link":He("error",t),He("load",t);case"area":case"base":case"br":case"col":case"hr":case"keygen":case"meta":case"param":case"track":case"wbr":case"menuitem":for(c in n)if(n.hasOwnProperty(c)&&(i=n[c],i!=null))switch(c){case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:ct(t,e,c,i,n,null)}return;default:if(Fp(e)){for(h in n)n.hasOwnProperty(h)&&(i=n[h],i!==void 0&&Ap(t,e,h,i,n,void 0));return}}for(o in n)n.hasOwnProperty(o)&&(i=n[o],i!=null&&ct(t,e,o,i,n,null))}function ME(t,e,n,i){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"input":var s=null,r=null,a=null,o=null,l=null,c=null,h=null;for(d in n){var p=n[d];if(n.hasOwnProperty(d)&&p!=null)switch(d){case"checked":break;case"value":break;case"defaultValue":l=p;default:i.hasOwnProperty(d)||ct(t,e,d,null,i,p)}}for(var u in i){var d=i[u];if(p=n[u],i.hasOwnProperty(u)&&(d!=null||p!=null))switch(u){case"type":r=d;break;case"name":s=d;break;case"checked":c=d;break;case"defaultChecked":h=d;break;case"value":a=d;break;case"defaultValue":o=d;break;case"children":case"dangerouslySetInnerHTML":if(d!=null)throw Error(J(137,e));break;default:d!==p&&ct(t,e,u,d,i,p)}}qd(t,a,o,l,c,h,r,s);return;case"select":d=a=o=u=null;for(r in n)if(l=n[r],n.hasOwnProperty(r)&&l!=null)switch(r){case"value":break;case"multiple":d=l;default:i.hasOwnProperty(r)||ct(t,e,r,null,i,l)}for(s in i)if(r=i[s],l=n[s],i.hasOwnProperty(s)&&(r!=null||l!=null))switch(s){case"value":u=r;break;case"defaultValue":o=r;break;case"multiple":a=r;default:r!==l&&ct(t,e,s,r,i,l)}e=o,n=a,i=d,u!=null?Ua(t,!!n,u,!1):!!i!=!!n&&(e!=null?Ua(t,!!n,e,!0):Ua(t,!!n,n?[]:"",!1));return;case"textarea":d=u=null;for(o in n)if(s=n[o],n.hasOwnProperty(o)&&s!=null&&!i.hasOwnProperty(o))switch(o){case"value":break;case"children":break;default:ct(t,e,o,null,i,s)}for(a in i)if(s=i[a],r=n[a],i.hasOwnProperty(a)&&(s!=null||r!=null))switch(a){case"value":u=s;break;case"defaultValue":d=s;break;case"children":break;case"dangerouslySetInnerHTML":if(s!=null)throw Error(J(91));break;default:s!==r&&ct(t,e,a,s,i,r)}Px(t,u,d);return;case"option":for(var v in n)u=n[v],n.hasOwnProperty(v)&&u!=null&&!i.hasOwnProperty(v)&&(v==="selected"?t.selected=!1:ct(t,e,v,null,i,u));for(l in i)u=i[l],d=n[l],i.hasOwnProperty(l)&&u!==d&&(u!=null||d!=null)&&(l==="selected"?t.selected=u&&typeof u!="function"&&typeof u!="symbol":ct(t,e,l,u,i,d));return;case"img":case"link":case"area":case"base":case"br":case"col":case"embed":case"hr":case"keygen":case"meta":case"param":case"source":case"track":case"wbr":case"menuitem":for(var M in n)u=n[M],n.hasOwnProperty(M)&&u!=null&&!i.hasOwnProperty(M)&&ct(t,e,M,null,i,u);for(c in i)if(u=i[c],d=n[c],i.hasOwnProperty(c)&&u!==d&&(u!=null||d!=null))switch(c){case"children":case"dangerouslySetInnerHTML":if(u!=null)throw Error(J(137,e));break;default:ct(t,e,c,u,i,d)}return;default:if(Fp(e)){for(var m in n)u=n[m],n.hasOwnProperty(m)&&u!==void 0&&!i.hasOwnProperty(m)&&Ap(t,e,m,void 0,i,u);for(h in i)u=i[h],d=n[h],!i.hasOwnProperty(h)||u===d||u===void 0&&d===void 0||Ap(t,e,h,u,i,d);return}}for(var f in n)u=n[f],n.hasOwnProperty(f)&&u!=null&&!i.hasOwnProperty(f)&&ct(t,e,f,null,i,u);for(p in i)u=i[p],d=n[p],!i.hasOwnProperty(p)||u===d||u==null&&d==null||ct(t,e,p,u,i,d)}function K0(t){switch(t){case"css":case"script":case"font":case"img":case"image":case"input":case"link":return!0;default:return!1}}function EE(){if(typeof performance.getEntriesByType=="function"){for(var t=0,e=0,n=performance.getEntriesByType("resource"),i=0;i<n.length;i++){var s=n[i],r=s.transferSize,a=s.initiatorType,o=s.duration;if(r&&o&&K0(a)){for(a=0,o=s.responseEnd,i+=1;i<n.length;i++){var l=n[i],c=l.startTime;if(c>o)break;var h=l.transferSize,p=l.initiatorType;h&&K0(p)&&(l=l.responseEnd,a+=h*(l<o?1:(o-c)/(l-c)))}if(--i,e+=8*(r+a)/(s.duration/1e3),t++,10<t)break}}if(0<t)return e/t/1e6}return navigator.connection&&(t=navigator.connection.downlink,typeof t=="number")?t:5}var Mp=null,Ep=null;function xu(t){return t.nodeType===9?t:t.ownerDocument}function J0(t){switch(t){case"http://www.w3.org/2000/svg":return 1;case"http://www.w3.org/1998/Math/MathML":return 2;default:return 0}}function G_(t,e){if(t===0)switch(e){case"svg":return 1;case"math":return 2;default:return 0}return t===1&&e==="foreignObject"?0:t}function Tp(t,e){return t==="textarea"||t==="noscript"||typeof e.children=="string"||typeof e.children=="number"||typeof e.children=="bigint"||typeof e.dangerouslySetInnerHTML=="object"&&e.dangerouslySetInnerHTML!==null&&e.dangerouslySetInnerHTML.__html!=null}var Ld=null;function TE(){var t=window.event;return t&&t.type==="popstate"?t===Ld?!1:(Ld=t,!0):(Ld=null,!1)}var H_=typeof setTimeout=="function"?setTimeout:void 0,bE=typeof clearTimeout=="function"?clearTimeout:void 0,j0=typeof Promise=="function"?Promise:void 0,wE=typeof queueMicrotask=="function"?queueMicrotask:typeof j0<"u"?function(t){return j0.resolve(null).then(t).catch(CE)}:H_;function CE(t){setTimeout(function(){throw t})}function ir(t){return t==="head"}function $0(t,e){var n=e,i=0;do{var s=n.nextSibling;if(t.removeChild(n),s&&s.nodeType===8)if(n=s.data,n==="/$"||n==="/&"){if(i===0){t.removeChild(s),qa(e);return}i--}else if(n==="$"||n==="$?"||n==="$~"||n==="$!"||n==="&")i++;else if(n==="html")tl(t.ownerDocument.documentElement);else if(n==="head"){n=t.ownerDocument.head,tl(n);for(var r=n.firstChild;r;){var a=r.nextSibling,o=r.nodeName;r[_l]||o==="SCRIPT"||o==="STYLE"||o==="LINK"&&r.rel.toLowerCase()==="stylesheet"||n.removeChild(r),r=a}}else n==="body"&&tl(t.ownerDocument.body);n=s}while(n);qa(e)}function ex(t,e){var n=t;t=0;do{var i=n.nextSibling;if(n.nodeType===1?e?(n._stashedDisplay=n.style.display,n.style.display="none"):(n.style.display=n._stashedDisplay||"",n.getAttribute("style")===""&&n.removeAttribute("style")):n.nodeType===3&&(e?(n._stashedText=n.nodeValue,n.nodeValue=""):n.nodeValue=n._stashedText||""),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(t===0)break;t--}else n!=="$"&&n!=="$?"&&n!=="$~"&&n!=="$!"||t++;n=i}while(n)}function bp(t){var e=t.firstChild;for(e&&e.nodeType===10&&(e=e.nextSibling);e;){var n=e;switch(e=e.nextSibling,n.nodeName){case"HTML":case"HEAD":case"BODY":bp(n),Op(n);continue;case"SCRIPT":case"STYLE":continue;case"LINK":if(n.rel.toLowerCase()==="stylesheet")continue}t.removeChild(n)}}function RE(t,e,n,i){for(;t.nodeType===1;){var s=n;if(t.nodeName.toLowerCase()!==e.toLowerCase()){if(!i&&(t.nodeName!=="INPUT"||t.type!=="hidden"))break}else if(i){if(!t[_l])switch(e){case"meta":if(!t.hasAttribute("itemprop"))break;return t;case"link":if(r=t.getAttribute("rel"),r==="stylesheet"&&t.hasAttribute("data-precedence"))break;if(r!==s.rel||t.getAttribute("href")!==(s.href==null||s.href===""?null:s.href)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin)||t.getAttribute("title")!==(s.title==null?null:s.title))break;return t;case"style":if(t.hasAttribute("data-precedence"))break;return t;case"script":if(r=t.getAttribute("src"),(r!==(s.src==null?null:s.src)||t.getAttribute("type")!==(s.type==null?null:s.type)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin))&&r&&t.hasAttribute("async")&&!t.hasAttribute("itemprop"))break;return t;default:return t}}else if(e==="input"&&t.type==="hidden"){var r=s.name==null?null:""+s.name;if(s.type==="hidden"&&t.getAttribute("name")===r)return t}else return t;if(t=di(t.nextSibling),t===null)break}return null}function DE(t,e,n){if(e==="")return null;for(;t.nodeType!==3;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!n||(t=di(t.nextSibling),t===null))return null;return t}function V_(t,e){for(;t.nodeType!==8;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!e||(t=di(t.nextSibling),t===null))return null;return t}function wp(t){return t.data==="$?"||t.data==="$~"}function Cp(t){return t.data==="$!"||t.data==="$?"&&t.ownerDocument.readyState!=="loading"}function UE(t,e){var n=t.ownerDocument;if(t.data==="$~")t._reactRetry=e;else if(t.data!=="$?"||n.readyState!=="loading")e();else{var i=function(){e(),n.removeEventListener("DOMContentLoaded",i)};n.addEventListener("DOMContentLoaded",i),t._reactRetry=i}}function di(t){for(;t!=null;t=t.nextSibling){var e=t.nodeType;if(e===1||e===3)break;if(e===8){if(e=t.data,e==="$"||e==="$!"||e==="$?"||e==="$~"||e==="&"||e==="F!"||e==="F")break;if(e==="/$"||e==="/&")return null}}return t}var Rp=null;function tx(t){t=t.nextSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="/$"||n==="/&"){if(e===0)return di(t.nextSibling);e--}else n!=="$"&&n!=="$!"&&n!=="$?"&&n!=="$~"&&n!=="&"||e++}t=t.nextSibling}return null}function nx(t){t=t.previousSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="$"||n==="$!"||n==="$?"||n==="$~"||n==="&"){if(e===0)return t;e--}else n!=="/$"&&n!=="/&"||e++}t=t.previousSibling}return null}function k_(t,e,n){switch(e=xu(n),t){case"html":if(t=e.documentElement,!t)throw Error(J(452));return t;case"head":if(t=e.head,!t)throw Error(J(453));return t;case"body":if(t=e.body,!t)throw Error(J(454));return t;default:throw Error(J(451))}}function tl(t){for(var e=t.attributes;e.length;)t.removeAttributeNode(e[0]);Op(t)}var pi=new Map,ix=new Set;function yu(t){return typeof t.getRootNode=="function"?t.getRootNode():t.nodeType===9?t:t.ownerDocument}var xs=it.d;it.d={f:BE,r:IE,D:PE,C:LE,L:NE,m:OE,X:zE,S:FE,M:GE};function BE(){var t=xs.f(),e=Nu();return t||e}function IE(t){var e=Za(t);e!==null&&e.tag===5&&e.type==="form"?Ny(e):xs.r(t)}var $a=typeof document>"u"?null:document;function W_(t,e,n){var i=$a;if(i&&typeof e=="string"&&e){var s=ci(e);s='link[rel="'+t+'"][href="'+s+'"]',typeof n=="string"&&(s+='[crossorigin="'+n+'"]'),ix.has(s)||(ix.add(s),t={rel:t,crossOrigin:n,href:e},i.querySelector(s)===null&&(e=i.createElement("link"),gn(e,"link",t),on(e),i.head.appendChild(e)))}}function PE(t){xs.D(t),W_("dns-prefetch",t,null)}function LE(t,e){xs.C(t,e),W_("preconnect",t,e)}function NE(t,e,n){xs.L(t,e,n);var i=$a;if(i&&t&&e){var s='link[rel="preload"][as="'+ci(e)+'"]';e==="image"&&n&&n.imageSrcSet?(s+='[imagesrcset="'+ci(n.imageSrcSet)+'"]',typeof n.imageSizes=="string"&&(s+='[imagesizes="'+ci(n.imageSizes)+'"]')):s+='[href="'+ci(t)+'"]';var r=s;switch(e){case"style":r=Ya(t);break;case"script":r=eo(t)}pi.has(r)||(t=Mt({rel:"preload",href:e==="image"&&n&&n.imageSrcSet?void 0:t,as:e},n),pi.set(r,t),i.querySelector(s)!==null||e==="style"&&i.querySelector(bl(r))||e==="script"&&i.querySelector(wl(r))||(e=i.createElement("link"),gn(e,"link",t),on(e),i.head.appendChild(e)))}}function OE(t,e){xs.m(t,e);var n=$a;if(n&&t){var i=e&&typeof e.as=="string"?e.as:"script",s='link[rel="modulepreload"][as="'+ci(i)+'"][href="'+ci(t)+'"]',r=s;switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":r=eo(t)}if(!pi.has(r)&&(t=Mt({rel:"modulepreload",href:t},e),pi.set(r,t),n.querySelector(s)===null)){switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":if(n.querySelector(wl(r)))return}i=n.createElement("link"),gn(i,"link",t),on(i),n.head.appendChild(i)}}}function FE(t,e,n){xs.S(t,e,n);var i=$a;if(i&&t){var s=Da(i).hoistableStyles,r=Ya(t);e=e||"default";var a=s.get(r);if(!a){var o={loading:0,preload:null};if(a=i.querySelector(bl(r)))o.loading=5;else{t=Mt({rel:"stylesheet",href:t,"data-precedence":e},n),(n=pi.get(r))&&Am(t,n);var l=a=i.createElement("link");on(l),gn(l,"link",t),l._p=new Promise(function(c,h){l.onload=c,l.onerror=h}),l.addEventListener("load",function(){o.loading|=1}),l.addEventListener("error",function(){o.loading|=2}),o.loading|=4,Xc(a,e,i)}a={type:"stylesheet",instance:a,count:1,state:o},s.set(r,a)}}}function zE(t,e){xs.X(t,e);var n=$a;if(n&&t){var i=Da(n).hoistableScripts,s=eo(t),r=i.get(s);r||(r=n.querySelector(wl(s)),r||(t=Mt({src:t,async:!0},e),(e=pi.get(s))&&Mm(t,e),r=n.createElement("script"),on(r),gn(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function GE(t,e){xs.M(t,e);var n=$a;if(n&&t){var i=Da(n).hoistableScripts,s=eo(t),r=i.get(s);r||(r=n.querySelector(wl(s)),r||(t=Mt({src:t,async:!0,type:"module"},e),(e=pi.get(s))&&Mm(t,e),r=n.createElement("script"),on(r),gn(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function sx(t,e,n,i){var s=(s=ks.current)?yu(s):null;if(!s)throw Error(J(446));switch(t){case"meta":case"title":return null;case"style":return typeof n.precedence=="string"&&typeof n.href=="string"?(e=Ya(n.href),n=Da(s).hoistableStyles,i=n.get(e),i||(i={type:"style",instance:null,count:0,state:null},n.set(e,i)),i):{type:"void",instance:null,count:0,state:null};case"link":if(n.rel==="stylesheet"&&typeof n.href=="string"&&typeof n.precedence=="string"){t=Ya(n.href);var r=Da(s).hoistableStyles,a=r.get(t);if(a||(s=s.ownerDocument||s,a={type:"stylesheet",instance:null,count:0,state:{loading:0,preload:null}},r.set(t,a),(r=s.querySelector(bl(t)))&&!r._p&&(a.instance=r,a.state.loading=5),pi.has(t)||(n={rel:"preload",as:"style",href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},pi.set(t,n),r||HE(s,t,n,a.state))),e&&i===null)throw Error(J(528,""));return a}if(e&&i!==null)throw Error(J(529,""));return null;case"script":return e=n.async,n=n.src,typeof n=="string"&&e&&typeof e!="function"&&typeof e!="symbol"?(e=eo(n),n=Da(s).hoistableScripts,i=n.get(e),i||(i={type:"script",instance:null,count:0,state:null},n.set(e,i)),i):{type:"void",instance:null,count:0,state:null};default:throw Error(J(444,t))}}function Ya(t){return'href="'+ci(t)+'"'}function bl(t){return'link[rel="stylesheet"]['+t+"]"}function X_(t){return Mt({},t,{"data-precedence":t.precedence,precedence:null})}function HE(t,e,n,i){t.querySelector('link[rel="preload"][as="style"]['+e+"]")?i.loading=1:(e=t.createElement("link"),i.preload=e,e.addEventListener("load",function(){return i.loading|=1}),e.addEventListener("error",function(){return i.loading|=2}),gn(e,"link",n),on(e),t.head.appendChild(e))}function eo(t){return'[src="'+ci(t)+'"]'}function wl(t){return"script[async]"+t}function rx(t,e,n){if(e.count++,e.instance===null)switch(e.type){case"style":var i=t.querySelector('style[data-href~="'+ci(n.href)+'"]');if(i)return e.instance=i,on(i),i;var s=Mt({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return i=(t.ownerDocument||t).createElement("style"),on(i),gn(i,"style",s),Xc(i,n.precedence,t),e.instance=i;case"stylesheet":s=Ya(n.href);var r=t.querySelector(bl(s));if(r)return e.state.loading|=4,e.instance=r,on(r),r;i=X_(n),(s=pi.get(s))&&Am(i,s),r=(t.ownerDocument||t).createElement("link"),on(r);var a=r;return a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),gn(r,"link",i),e.state.loading|=4,Xc(r,n.precedence,t),e.instance=r;case"script":return r=eo(n.src),(s=t.querySelector(wl(r)))?(e.instance=s,on(s),s):(i=n,(s=pi.get(r))&&(i=Mt({},n),Mm(i,s)),t=t.ownerDocument||t,s=t.createElement("script"),on(s),gn(s,"link",i),t.head.appendChild(s),e.instance=s);case"void":return null;default:throw Error(J(443,e.type))}else e.type==="stylesheet"&&(e.state.loading&4)===0&&(i=e.instance,e.state.loading|=4,Xc(i,n.precedence,t));return e.instance}function Xc(t,e,n){for(var i=n.querySelectorAll('link[rel="stylesheet"][data-precedence],style[data-precedence]'),s=i.length?i[i.length-1]:null,r=s,a=0;a<i.length;a++){var o=i[a];if(o.dataset.precedence===e)r=o;else if(r!==s)break}r?r.parentNode.insertBefore(t,r.nextSibling):(e=n.nodeType===9?n.head:n,e.insertBefore(t,e.firstChild))}function Am(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.title==null&&(t.title=e.title)}function Mm(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.integrity==null&&(t.integrity=e.integrity)}var Yc=null;function ax(t,e,n){if(Yc===null){var i=new Map,s=Yc=new Map;s.set(n,i)}else s=Yc,i=s.get(n),i||(i=new Map,s.set(n,i));if(i.has(t))return i;for(i.set(t,null),n=n.getElementsByTagName(t),s=0;s<n.length;s++){var r=n[s];if(!(r[_l]||r[dn]||t==="link"&&r.getAttribute("rel")==="stylesheet")&&r.namespaceURI!=="http://www.w3.org/2000/svg"){var a=r.getAttribute(e)||"";a=t+a;var o=i.get(a);o?o.push(r):i.set(a,[r])}}return i}function ox(t,e,n){t=t.ownerDocument||t,t.head.insertBefore(n,e==="title"?t.querySelector("head > title"):null)}function VE(t,e,n){if(n===1||e.itemProp!=null)return!1;switch(t){case"meta":case"title":return!0;case"style":if(typeof e.precedence!="string"||typeof e.href!="string"||e.href==="")break;return!0;case"link":if(typeof e.rel!="string"||typeof e.href!="string"||e.href===""||e.onLoad||e.onError)break;return e.rel==="stylesheet"?(t=e.disabled,typeof e.precedence=="string"&&t==null):!0;case"script":if(e.async&&typeof e.async!="function"&&typeof e.async!="symbol"&&!e.onLoad&&!e.onError&&e.src&&typeof e.src=="string")return!0}return!1}function Y_(t){return!(t.type==="stylesheet"&&(t.state.loading&3)===0)}function kE(t,e,n,i){if(n.type==="stylesheet"&&(typeof i.media!="string"||matchMedia(i.media).matches!==!1)&&(n.state.loading&4)===0){if(n.instance===null){var s=Ya(i.href),r=e.querySelector(bl(s));if(r){e=r._p,e!==null&&typeof e=="object"&&typeof e.then=="function"&&(t.count++,t=_u.bind(t),e.then(t,t)),n.state.loading|=4,n.instance=r,on(r);return}r=e.ownerDocument||e,i=X_(i),(s=pi.get(s))&&Am(i,s),r=r.createElement("link"),on(r);var a=r;a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),gn(r,"link",i),n.instance=r}t.stylesheets===null&&(t.stylesheets=new Map),t.stylesheets.set(n,e),(e=n.state.preload)&&(n.state.loading&3)===0&&(t.count++,n=_u.bind(t),e.addEventListener("load",n),e.addEventListener("error",n))}}var Nd=0;function WE(t,e){return t.stylesheets&&t.count===0&&qc(t,t.stylesheets),0<t.count||0<t.imgCount?function(n){var i=setTimeout(function(){if(t.stylesheets&&qc(t,t.stylesheets),t.unsuspend){var r=t.unsuspend;t.unsuspend=null,r()}},6e4+e);0<t.imgBytes&&Nd===0&&(Nd=62500*EE());var s=setTimeout(function(){if(t.waitingForImages=!1,t.count===0&&(t.stylesheets&&qc(t,t.stylesheets),t.unsuspend)){var r=t.unsuspend;t.unsuspend=null,r()}},(t.imgBytes>Nd?50:800)+e);return t.unsuspend=n,function(){t.unsuspend=null,clearTimeout(i),clearTimeout(s)}}:null}function _u(){if(this.count--,this.count===0&&(this.imgCount===0||!this.waitingForImages)){if(this.stylesheets)qc(this,this.stylesheets);else if(this.unsuspend){var t=this.unsuspend;this.unsuspend=null,t()}}}var Su=null;function qc(t,e){t.stylesheets=null,t.unsuspend!==null&&(t.count++,Su=new Map,e.forEach(XE,t),Su=null,_u.call(t))}function XE(t,e){if(!(e.state.loading&4)){var n=Su.get(t);if(n)var i=n.get(null);else{n=new Map,Su.set(t,n);for(var s=t.querySelectorAll("link[data-precedence],style[data-precedence]"),r=0;r<s.length;r++){var a=s[r];(a.nodeName==="LINK"||a.getAttribute("media")!=="not all")&&(n.set(a.dataset.precedence,a),i=a)}i&&n.set(null,i)}s=e.instance,a=s.getAttribute("data-precedence"),r=n.get(a)||i,r===i&&n.set(null,s),n.set(a,s),this.count++,i=_u.bind(this),s.addEventListener("load",i),s.addEventListener("error",i),r?r.parentNode.insertBefore(s,r.nextSibling):(t=t.nodeType===9?t.head:t,t.insertBefore(s,t.firstChild)),e.state.loading|=4}}var dl={$$typeof:os,Provider:null,Consumer:null,_currentValue:Ur,_currentValue2:Ur,_threadCount:0};function YE(t,e,n,i,s,r,a,o,l){this.tag=1,this.containerInfo=t,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=ld(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=ld(0),this.hiddenUpdates=ld(null),this.identifierPrefix=i,this.onUncaughtError=s,this.onCaughtError=r,this.onRecoverableError=a,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=l,this.incompleteTransitions=new Map}function q_(t,e,n,i,s,r,a,o,l,c,h,p){return t=new YE(t,e,n,a,l,c,h,p,o),e=1,r===!0&&(e|=24),r=Wn(3,null,null,e),t.current=r,r.stateNode=t,e=Zp(),e.refCount++,t.pooledCache=e,e.refCount++,r.memoizedState={element:i,isDehydrated:n,cache:e},jp(r),t}function Q_(t){return t?(t=ba,t):ba}function Z_(t,e,n,i,s,r){s=Q_(s),i.context===null?i.context=s:i.pendingContext=s,i=Xs(e),i.payload={element:n},r=r===void 0?null:r,r!==null&&(i.callback=r),n=Ys(t,i,e),n!==null&&(Ln(n,t,e),qo(n,t,e))}function lx(t,e){if(t=t.memoizedState,t!==null&&t.dehydrated!==null){var n=t.retryLane;t.retryLane=n!==0&&n<e?n:e}}function Em(t,e){lx(t,e),(t=t.alternate)&&lx(t,e)}function K_(t){if(t.tag===13||t.tag===31){var e=Wr(t,67108864);e!==null&&Ln(e,t,67108864),Em(t,67108864)}}function cx(t){if(t.tag===13||t.tag===31){var e=Zn();e=Lp(e);var n=Wr(t,e);n!==null&&Ln(n,t,e),Em(t,e)}}var Au=!0;function qE(t,e,n,i){var s=Be.T;Be.T=null;var r=it.p;try{it.p=2,Tm(t,e,n,i)}finally{it.p=r,Be.T=s}}function QE(t,e,n,i){var s=Be.T;Be.T=null;var r=it.p;try{it.p=8,Tm(t,e,n,i)}finally{it.p=r,Be.T=s}}function Tm(t,e,n,i){if(Au){var s=Dp(i);if(s===null)Pd(t,e,i,Mu,n),ux(t,i);else if(KE(s,t,e,n,i))i.stopPropagation();else if(ux(t,i),e&4&&-1<ZE.indexOf(t)){for(;s!==null;){var r=Za(s);if(r!==null)switch(r.tag){case 3:if(r=r.stateNode,r.current.memoizedState.isDehydrated){var a=Cr(r.pendingLanes);if(a!==0){var o=r;for(o.pendingLanes|=2,o.entangledLanes|=2;a;){var l=1<<31-Qn(a);o.entanglements[1]|=l,a&=~l}Vi(r),(nt&6)===0&&(hu=Yn()+500,Tl(0,!1))}}break;case 31:case 13:o=Wr(r,2),o!==null&&Ln(o,r,2),Nu(),Em(r,2)}if(r=Dp(i),r===null&&Pd(t,e,i,Mu,n),r===s)break;s=r}s!==null&&i.stopPropagation()}else Pd(t,e,i,null,n)}}function Dp(t){return t=zp(t),bm(t)}var Mu=null;function bm(t){if(Mu=null,t=_a(t),t!==null){var e=gl(t);if(e===null)t=null;else{var n=e.tag;if(n===13){if(t=gx(e),t!==null)return t;t=null}else if(n===31){if(t=vx(e),t!==null)return t;t=null}else if(n===3){if(e.stateNode.current.memoizedState.isDehydrated)return e.tag===3?e.stateNode.containerInfo:null;t=null}else e!==t&&(t=null)}}return Mu=t,null}function J_(t){switch(t){case"beforetoggle":case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"toggle":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 2;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 8;case"message":switch(N1()){case Sx:return 2;case Ax:return 8;case jc:case O1:return 32;case Mx:return 268435456;default:return 32}default:return 32}}var Up=!1,Zs=null,Ks=null,Js=null,pl=new Map,ml=new Map,Os=[],ZE="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(" ");function ux(t,e){switch(t){case"focusin":case"focusout":Zs=null;break;case"dragenter":case"dragleave":Ks=null;break;case"mouseover":case"mouseout":Js=null;break;case"pointerover":case"pointerout":pl.delete(e.pointerId);break;case"gotpointercapture":case"lostpointercapture":ml.delete(e.pointerId)}}function No(t,e,n,i,s,r){return t===null||t.nativeEvent!==r?(t={blockedOn:e,domEventName:n,eventSystemFlags:i,nativeEvent:r,targetContainers:[s]},e!==null&&(e=Za(e),e!==null&&K_(e)),t):(t.eventSystemFlags|=i,e=t.targetContainers,s!==null&&e.indexOf(s)===-1&&e.push(s),t)}function KE(t,e,n,i,s){switch(e){case"focusin":return Zs=No(Zs,t,e,n,i,s),!0;case"dragenter":return Ks=No(Ks,t,e,n,i,s),!0;case"mouseover":return Js=No(Js,t,e,n,i,s),!0;case"pointerover":var r=s.pointerId;return pl.set(r,No(pl.get(r)||null,t,e,n,i,s)),!0;case"gotpointercapture":return r=s.pointerId,ml.set(r,No(ml.get(r)||null,t,e,n,i,s)),!0}return!1}function j_(t){var e=_a(t.target);if(e!==null){var n=gl(e);if(n!==null){if(e=n.tag,e===13){if(e=gx(n),e!==null){t.blockedOn=e,qv(t.priority,function(){cx(n)});return}}else if(e===31){if(e=vx(n),e!==null){t.blockedOn=e,qv(t.priority,function(){cx(n)});return}}else if(e===3&&n.stateNode.current.memoizedState.isDehydrated){t.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}t.blockedOn=null}function Qc(t){if(t.blockedOn!==null)return!1;for(var e=t.targetContainers;0<e.length;){var n=Dp(t.nativeEvent);if(n===null){n=t.nativeEvent;var i=new n.constructor(n.type,n);Zd=i,n.target.dispatchEvent(i),Zd=null}else return e=Za(n),e!==null&&K_(e),t.blockedOn=n,!1;e.shift()}return!0}function fx(t,e,n){Qc(t)&&n.delete(e)}function JE(){Up=!1,Zs!==null&&Qc(Zs)&&(Zs=null),Ks!==null&&Qc(Ks)&&(Ks=null),Js!==null&&Qc(Js)&&(Js=null),pl.forEach(fx),ml.forEach(fx)}function Bc(t,e){t.blockedOn===e&&(t.blockedOn=null,Up||(Up=!0,tn.unstable_scheduleCallback(tn.unstable_NormalPriority,JE)))}var Ic=null;function hx(t){Ic!==t&&(Ic=t,tn.unstable_scheduleCallback(tn.unstable_NormalPriority,function(){Ic===t&&(Ic=null);for(var e=0;e<t.length;e+=3){var n=t[e],i=t[e+1],s=t[e+2];if(typeof i!="function"){if(bm(i||n)===null)continue;break}var r=Za(n);r!==null&&(t.splice(e,3),e-=3,fp(r,{pending:!0,data:s,method:n.method,action:i},i,s))}}))}function qa(t){function e(l){return Bc(l,t)}Zs!==null&&Bc(Zs,t),Ks!==null&&Bc(Ks,t),Js!==null&&Bc(Js,t),pl.forEach(e),ml.forEach(e);for(var n=0;n<Os.length;n++){var i=Os[n];i.blockedOn===t&&(i.blockedOn=null)}for(;0<Os.length&&(n=Os[0],n.blockedOn===null);)j_(n),n.blockedOn===null&&Os.shift();if(n=(t.ownerDocument||t).$$reactFormReplay,n!=null)for(i=0;i<n.length;i+=3){var s=n[i],r=n[i+1],a=s[Nn]||null;if(typeof r=="function")a||hx(n);else if(a){var o=null;if(r&&r.hasAttribute("formAction")){if(s=r,a=r[Nn]||null)o=a.formAction;else if(bm(s)!==null)continue}else o=a.action;typeof o=="function"?n[i+1]=o:(n.splice(i,3),i-=3),hx(n)}}}function $_(){function t(r){r.canIntercept&&r.info==="react-transition"&&r.intercept({handler:function(){return new Promise(function(a){return s=a})},focusReset:"manual",scroll:"manual"})}function e(){s!==null&&(s(),s=null),i||setTimeout(n,20)}function n(){if(!i&&!navigation.transition){var r=navigation.currentEntry;r&&r.url!=null&&navigation.navigate(r.url,{state:r.getState(),info:"react-transition",history:"replace"})}}if(typeof navigation=="object"){var i=!1,s=null;return navigation.addEventListener("navigate",t),navigation.addEventListener("navigatesuccess",e),navigation.addEventListener("navigateerror",e),setTimeout(n,100),function(){i=!0,navigation.removeEventListener("navigate",t),navigation.removeEventListener("navigatesuccess",e),navigation.removeEventListener("navigateerror",e),s!==null&&(s(),s=null)}}}function wm(t){this._internalRoot=t}zu.prototype.render=wm.prototype.render=function(t){var e=this._internalRoot;if(e===null)throw Error(J(409));var n=e.current,i=Zn();Z_(n,i,t,e,null,null)};zu.prototype.unmount=wm.prototype.unmount=function(){var t=this._internalRoot;if(t!==null){this._internalRoot=null;var e=t.containerInfo;Z_(t.current,2,null,t,null,null),Nu(),e[Qa]=null}};function zu(t){this._internalRoot=t}zu.prototype.unstable_scheduleHydration=function(t){if(t){var e=Cx();t={blockedOn:null,target:t,priority:e};for(var n=0;n<Os.length&&e!==0&&e<Os[n].priority;n++);Os.splice(n,0,t),n===0&&j_(t)}};var dx=px.version;if(dx!=="19.2.8")throw Error(J(527,dx,"19.2.8"));it.findDOMNode=function(t){var e=t._reactInternals;if(e===void 0)throw typeof t.render=="function"?Error(J(188)):(t=Object.keys(t).join(","),Error(J(268,t)));return t=R1(e),t=t!==null?xx(t):null,t=t===null?null:t.stateNode,t};var jE={bundleType:0,version:"19.2.8",rendererPackageName:"react-dom",currentDispatcherRef:Be,reconcilerVersion:"19.2.8"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"&&(Oo=__REACT_DEVTOOLS_GLOBAL_HOOK__,!Oo.isDisabled&&Oo.supportsFiber))try{vl=Oo.inject(jE),qn=Oo}catch{}var Oo;Gu.createRoot=function(t,e){if(!mx(t))throw Error(J(299));var n=!1,i="",s=Wy,r=Xy,a=Yy;return e!=null&&(e.unstable_strictMode===!0&&(n=!0),e.identifierPrefix!==void 0&&(i=e.identifierPrefix),e.onUncaughtError!==void 0&&(s=e.onUncaughtError),e.onCaughtError!==void 0&&(r=e.onCaughtError),e.onRecoverableError!==void 0&&(a=e.onRecoverableError)),e=q_(t,1,!1,null,null,n,i,null,s,r,a,$_),t[Qa]=e.current,Sm(t),new wm(e)};Gu.hydrateRoot=function(t,e,n){if(!mx(t))throw Error(J(299));var i=!1,s="",r=Wy,a=Xy,o=Yy,l=null;return n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(s=n.identifierPrefix),n.onUncaughtError!==void 0&&(r=n.onUncaughtError),n.onCaughtError!==void 0&&(a=n.onCaughtError),n.onRecoverableError!==void 0&&(o=n.onRecoverableError),n.formState!==void 0&&(l=n.formState)),e=q_(t,1,!0,e,n??null,i,s,l,r,a,o,$_),e.context=Q_(null),n=e.current,i=Zn(),i=Lp(i),s=Xs(i),s.callback=null,Ys(n,s,i),n=i,e.current.lanes=n,yl(e,n),Vi(e),t[Qa]=e.current,Sm(t),new zu(e)};Gu.version="19.2.8"});var iS=Li((DD,nS)=>{"use strict";function tS(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(tS)}catch(t){console.error(t)}}tS(),nS.exports=eS()});var WA=Li(Fh=>{"use strict";var pD=Symbol.for("react.transitional.element"),mD=Symbol.for("react.fragment");function kA(t,e,n){var i=null;if(n!==void 0&&(i=""+n),e.key!==void 0&&(i=""+e.key),"key"in e){n={};for(var s in e)s!=="key"&&(n[s]=e[s])}else n=e;return e=n.ref,{$$typeof:pD,type:t,key:i,ref:e!==void 0?e:null,props:n}}Fh.Fragment=mD;Fh.jsx=kA;Fh.jsxs=kA});var lc=Li((sP,XA)=>{"use strict";XA.exports=WA()});var Gh=Tr(bo()),QA=Tr(iS());var AS=0,rg=1,MS=2;var Ql=1,ES=2,So=3,wi=0,Jt=1,Rn=2,Gn=0,$r=1,ag=2,og=3,lg=4,TS=5;var fr=100,bS=101,wS=102,CS=103,RS=104,DS=200,US=201,BS=202,IS=203,cf=204,uf=205,PS=206,LS=207,NS=208,OS=209,FS=210,zS=211,GS=212,HS=213,VS=214,ff=0,mo=1,hf=2,ea=3,df=4,pf=5,mf=6,gf=7,cg=0,kS=1,WS=2,Di=0,ug=1,fg=2,hg=3,dg=4,pg=5,mg=6,gg=7;var vg=300,xr=301,na=302,kf=303,Wf=304,Zl=306,vf=1e3,Xi=1001,xf=1002,un=1003,XS=1004;var Kl=1005;var xt=1006,Xf=1007;var yr=1008;var jt=1009,xg=1010,yg=1011,Ao=1012,Yf=1013,Ui=1014,ti=1015,Qi=1016,qf=1017,Qf=1018,_r=1020,_g=35902,Sg=35899,Ag=1021,Mg=1022,yi=1023,Yi=1026,Zi=1027,Eg=1028,Zf=1029,Sr=1030,Kf=1031;var Jf=1033,Jl=33776,jl=33777,$l=33778,ec=33779,jf=35840,$f=35841,eh=35842,th=35843,nh=36196,ih=37492,sh=37496,rh=37488,ah=37489,tc=37490,oh=37491,lh=37808,ch=37809,uh=37810,fh=37811,hh=37812,dh=37813,ph=37814,mh=37815,gh=37816,vh=37817,xh=37818,yh=37819,_h=37820,Sh=37821,Ah=36492,Mh=36494,Eh=36495,Th=36283,bh=36284,nc=36285,wh=36286;var Il=2300,yf=2301,lf=2302,Jm=2303,jm=2400,$m=2401,eg=2402;var Ki=3200;var Tg=0,YS=1,ni="",Tt="srgb",Ts="srgb-linear",Pl="linear",st="srgb";var Kr=7680;var tg=519,qS=512,QS=513,ZS=514,Ch=515,KS=516,JS=517,Rh=518,jS=519,ng=35044;var ic="300 es",bi=2e3,Ll=2001;function $E(t){for(let e=t.length-1;e>=0;--e)if(t[e]>=65535)return!0;return!1}function eT(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)}function Nl(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function $S(){let t=Nl("canvas");return t.style.display="block",t}var sS={},go=null;function bg(...t){let e="THREE."+t.shift();go?go("log",e,...t):console.log(e,...t)}function eA(t){let e=t[0];if(typeof e=="string"&&e.startsWith("TSL:")){let n=t[1];n&&n.isStackTrace?t[0]+=" "+n.getLocation():t[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return t}function Re(...t){t=eA(t);let e="THREE."+t.shift();if(go)go("warn",e,...t);else{let n=t[0];n&&n.isStackTrace?console.warn(n.getError(e)):console.warn(e,...t)}}function Ue(...t){t=eA(t);let e="THREE."+t.shift();if(go)go("error",e,...t);else{let n=t[0];n&&n.isStackTrace?console.error(n.getError(e)):console.error(e,...t)}}function jr(...t){let e=t.join(" ");e in sS||(sS[e]=!0,Re(...t))}function tA(t,e,n){return new Promise(function(i,s){function r(){switch(t.clientWaitSync(e,t.SYNC_FLUSH_COMMANDS_BIT,0)){case t.WAIT_FAILED:s();break;case t.TIMEOUT_EXPIRED:setTimeout(r,n);break;default:i()}}setTimeout(r,n)})}var nA={[ff]:mo,[hf]:mf,[df]:gf,[ea]:pf,[mo]:ff,[mf]:hf,[gf]:df,[pf]:ea},zn=class{addEventListener(e,n){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(n)===-1&&i[e].push(n)}hasEventListener(e,n){let i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(n)!==-1}removeEventListener(e,n){let i=this._listeners;if(i===void 0)return;let s=i[e];if(s!==void 0){let r=s.indexOf(n);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let n=this._listeners;if(n===void 0)return;let i=n[e.type];if(i!==void 0){e.target=this;let s=i.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,e);e.target=null}}},yn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];var Cm=Math.PI/180,_f=180/Math.PI;function sc(){let t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(yn[t&255]+yn[t>>8&255]+yn[t>>16&255]+yn[t>>24&255]+"-"+yn[e&255]+yn[e>>8&255]+"-"+yn[e>>16&15|64]+yn[e>>24&255]+"-"+yn[n&63|128]+yn[n>>8&255]+"-"+yn[n>>16&255]+yn[n>>24&255]+yn[i&255]+yn[i>>8&255]+yn[i>>16&255]+yn[i>>24&255]).toLowerCase()}function Qe(t,e,n){return Math.max(e,Math.min(n,t))}function tT(t,e){return(t%e+e)%e}function Rm(t,e,n){return(1-n)*t+n*e}function Cl(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return t/4294967295;case Uint16Array:return t/65535;case Uint8Array:return t/255;case Int32Array:return Math.max(t/2147483647,-1);case Int16Array:return Math.max(t/32767,-1);case Int8Array:return Math.max(t/127,-1);default:throw new Error("THREE.MathUtils: Invalid component type.")}}function Fn(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return Math.round(t*4294967295);case Uint16Array:return Math.round(t*65535);case Uint8Array:return Math.round(t*255);case Int32Array:return Math.round(t*2147483647);case Int16Array:return Math.round(t*32767);case Int8Array:return Math.round(t*127);default:throw new Error("THREE.MathUtils: Invalid component type.")}}var Ug=class Ug{constructor(e=0,n=0){this.x=e,this.y=n}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,n){return this.x=e,this.y=n,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let n=this.x,i=this.y,s=e.elements;return this.x=s[0]*n+s[3]*i+s[6],this.y=s[1]*n+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,n){return this.x=Qe(this.x,e.x,n.x),this.y=Qe(this.y,e.y,n.y),this}clampScalar(e,n){return this.x=Qe(this.x,e,n),this.y=Qe(this.y,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Qe(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos(Qe(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y;return n*n+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this}rotateAround(e,n){let i=Math.cos(n),s=Math.sin(n),r=this.x-e.x,a=this.y-e.y;return this.x=r*i-a*s+e.x,this.y=r*s+a*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};Ug.prototype.isVector2=!0;var Ie=Ug,qi=class{constructor(e=0,n=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=n,this._z=i,this._w=s}static slerpFlat(e,n,i,s,r,a,o){let l=i[s+0],c=i[s+1],h=i[s+2],p=i[s+3],u=r[a+0],d=r[a+1],v=r[a+2],M=r[a+3];if(p!==M||l!==u||c!==d||h!==v){let m=l*u+c*d+h*v+p*M;m<0&&(u=-u,d=-d,v=-v,M=-M,m=-m);let f=1-o;if(m<.9995){let g=Math.acos(m),S=Math.sin(g);f=Math.sin(f*g)/S,o=Math.sin(o*g)/S,l=l*f+u*o,c=c*f+d*o,h=h*f+v*o,p=p*f+M*o}else{l=l*f+u*o,c=c*f+d*o,h=h*f+v*o,p=p*f+M*o;let g=1/Math.sqrt(l*l+c*c+h*h+p*p);l*=g,c*=g,h*=g,p*=g}}e[n]=l,e[n+1]=c,e[n+2]=h,e[n+3]=p}static multiplyQuaternionsFlat(e,n,i,s,r,a){let o=i[s],l=i[s+1],c=i[s+2],h=i[s+3],p=r[a],u=r[a+1],d=r[a+2],v=r[a+3];return e[n]=o*v+h*p+l*d-c*u,e[n+1]=l*v+h*u+c*p-o*d,e[n+2]=c*v+h*d+o*u-l*p,e[n+3]=h*v-o*p-l*u-c*d,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,n,i,s){return this._x=e,this._y=n,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,n=!0){let i=e._x,s=e._y,r=e._z,a=e._order,o=Math.cos,l=Math.sin,c=o(i/2),h=o(s/2),p=o(r/2),u=l(i/2),d=l(s/2),v=l(r/2);switch(a){case"XYZ":this._x=u*h*p+c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p-u*d*v;break;case"YXZ":this._x=u*h*p+c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p+u*d*v;break;case"ZXY":this._x=u*h*p-c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p-u*d*v;break;case"ZYX":this._x=u*h*p-c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p+u*d*v;break;case"YZX":this._x=u*h*p+c*d*v,this._y=c*d*p+u*h*v,this._z=c*h*v-u*d*p,this._w=c*h*p-u*d*v;break;case"XZY":this._x=u*h*p-c*d*v,this._y=c*d*p-u*h*v,this._z=c*h*v+u*d*p,this._w=c*h*p+u*d*v;break;default:Re("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return n===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,n){let i=n/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){let n=e.elements,i=n[0],s=n[4],r=n[8],a=n[1],o=n[5],l=n[9],c=n[2],h=n[6],p=n[10],u=i+o+p;if(u>0){let d=.5/Math.sqrt(u+1);this._w=.25/d,this._x=(h-l)*d,this._y=(r-c)*d,this._z=(a-s)*d}else if(i>o&&i>p){let d=2*Math.sqrt(1+i-o-p);this._w=(h-l)/d,this._x=.25*d,this._y=(s+a)/d,this._z=(r+c)/d}else if(o>p){let d=2*Math.sqrt(1+o-i-p);this._w=(r-c)/d,this._x=(s+a)/d,this._y=.25*d,this._z=(l+h)/d}else{let d=2*Math.sqrt(1+p-i-o);this._w=(a-s)/d,this._x=(r+c)/d,this._y=(l+h)/d,this._z=.25*d}return this._onChangeCallback(),this}setFromUnitVectors(e,n){let i=e.dot(n)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*n.z-e.z*n.y,this._y=e.z*n.x-e.x*n.z,this._z=e.x*n.y-e.y*n.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Qe(this.dot(e),-1,1)))}rotateTowards(e,n){let i=this.angleTo(e);if(i===0)return this;let s=Math.min(1,n/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=n._x,l=n._y,c=n._z,h=n._w;return this._x=i*h+a*o+s*c-r*l,this._y=s*h+a*l+r*o-i*c,this._z=r*h+a*c+i*l-s*o,this._w=a*h-i*o-s*l-r*c,this._onChangeCallback(),this}slerp(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=this.dot(e);o<0&&(i=-i,s=-s,r=-r,a=-a,o=-o);let l=1-n;if(o<.9995){let c=Math.acos(o),h=Math.sin(c);l=Math.sin(l*c)/h,n=Math.sin(n*c)/h,this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this._onChangeCallback()}else this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this.normalize();return this}slerpQuaternions(e,n,i){return this.copy(e).slerp(n,i)}random(){let e=2*Math.PI*Math.random(),n=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(n),r*Math.cos(n))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,n=0){return this._x=e[n],this._y=e[n+1],this._z=e[n+2],this._w=e[n+3],this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._w,e}fromBufferAttribute(e,n){return this._x=e.getX(n),this._y=e.getY(n),this._z=e.getZ(n),this._w=e.getW(n),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},Bg=class Bg{constructor(e=0,n=0,i=0){this.x=e,this.y=n,this.z=i}set(e,n,i){return i===void 0&&(i=this.z),this.x=e,this.y=n,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,n){return this.x=e.x*n.x,this.y=e.y*n.y,this.z=e.z*n.z,this}applyEuler(e){return this.applyQuaternion(rS.setFromEuler(e))}applyAxisAngle(e,n){return this.applyQuaternion(rS.setFromAxisAngle(e,n))}applyMatrix3(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[3]*i+r[6]*s,this.y=r[1]*n+r[4]*i+r[7]*s,this.z=r[2]*n+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=e.elements,a=1/(r[3]*n+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*n+r[4]*i+r[8]*s+r[12])*a,this.y=(r[1]*n+r[5]*i+r[9]*s+r[13])*a,this.z=(r[2]*n+r[6]*i+r[10]*s+r[14])*a,this}applyQuaternion(e){let n=this.x,i=this.y,s=this.z,r=e.x,a=e.y,o=e.z,l=e.w,c=2*(a*s-o*i),h=2*(o*n-r*s),p=2*(r*i-a*n);return this.x=n+l*c+a*p-o*h,this.y=i+l*h+o*c-r*p,this.z=s+l*p+r*h-a*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[4]*i+r[8]*s,this.y=r[1]*n+r[5]*i+r[9]*s,this.z=r[2]*n+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,n){return this.x=Qe(this.x,e.x,n.x),this.y=Qe(this.y,e.y,n.y),this.z=Qe(this.z,e.z,n.z),this}clampScalar(e,n){return this.x=Qe(this.x,e,n),this.y=Qe(this.y,e,n),this.z=Qe(this.z,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Qe(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,n){let i=e.x,s=e.y,r=e.z,a=n.x,o=n.y,l=n.z;return this.x=s*l-r*o,this.y=r*a-i*l,this.z=i*o-s*a,this}projectOnVector(e){let n=e.lengthSq();if(n===0)return this.set(0,0,0);let i=e.dot(this)/n;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return Dm.copy(this).projectOnVector(e),this.sub(Dm)}reflect(e){return this.sub(Dm.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos(Qe(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return n*n+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,n,i){let s=Math.sin(n)*e;return this.x=s*Math.sin(i),this.y=Math.cos(n)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,n,i){return this.x=e*Math.sin(n),this.y=i,this.z=e*Math.cos(n),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this}setFromMatrixScale(e){let n=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=n,this.y=i,this.z=s,this}setFromMatrixColumn(e,n){return this.fromArray(e.elements,n*4)}setFromMatrix3Column(e,n){return this.fromArray(e.elements,n*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,n=Math.random()*2-1,i=Math.sqrt(1-n*n);return this.x=i*Math.cos(e),this.y=n,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};Bg.prototype.isVector3=!0;var z=Bg,Dm=new z,rS=new qi,Ig=class Ig{constructor(e,n,i,s,r,a,o,l,c){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c)}set(e,n,i,s,r,a,o,l,c){let h=this.elements;return h[0]=e,h[1]=s,h[2]=o,h[3]=n,h[4]=r,h[5]=l,h[6]=i,h[7]=a,h[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],this}extractBasis(e,n,i){return e.setFromMatrix3Column(this,0),n.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let n=e.elements;return this.set(n[0],n[4],n[8],n[1],n[5],n[9],n[2],n[6],n[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[3],l=i[6],c=i[1],h=i[4],p=i[7],u=i[2],d=i[5],v=i[8],M=s[0],m=s[3],f=s[6],g=s[1],S=s[4],_=s[7],T=s[2],b=s[5],w=s[8];return r[0]=a*M+o*g+l*T,r[3]=a*m+o*S+l*b,r[6]=a*f+o*_+l*w,r[1]=c*M+h*g+p*T,r[4]=c*m+h*S+p*b,r[7]=c*f+h*_+p*w,r[2]=u*M+d*g+v*T,r[5]=u*m+d*S+v*b,r[8]=u*f+d*_+v*w,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[3]*=e,n[6]*=e,n[1]*=e,n[4]*=e,n[7]*=e,n[2]*=e,n[5]*=e,n[8]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8];return n*a*h-n*o*c-i*r*h+i*o*l+s*r*c-s*a*l}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=h*a-o*c,u=o*l-h*r,d=c*r-a*l,v=n*p+i*u+s*d;if(v===0)return this.set(0,0,0,0,0,0,0,0,0);let M=1/v;return e[0]=p*M,e[1]=(s*c-h*i)*M,e[2]=(o*i-s*a)*M,e[3]=u*M,e[4]=(h*n-s*l)*M,e[5]=(s*r-o*n)*M,e[6]=d*M,e[7]=(i*l-c*n)*M,e[8]=(a*n-i*r)*M,this}transpose(){let e,n=this.elements;return e=n[1],n[1]=n[3],n[3]=e,e=n[2],n[2]=n[6],n[6]=e,e=n[5],n[5]=n[7],n[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let n=this.elements;return e[0]=n[0],e[1]=n[3],e[2]=n[6],e[3]=n[1],e[4]=n[4],e[5]=n[7],e[6]=n[2],e[7]=n[5],e[8]=n[8],this}setUvTransform(e,n,i,s,r,a,o){let l=Math.cos(r),c=Math.sin(r);return this.set(i*l,i*c,-i*(l*a+c*o)+a+e,-s*c,s*l,-s*(-c*a+l*o)+o+n,0,0,1),this}scale(e,n){return jr("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(Um.makeScale(e,n)),this}rotate(e){return jr("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(Um.makeRotation(-e)),this}translate(e,n){return jr("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(Um.makeTranslation(e,n)),this}makeTranslation(e,n){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,n,0,0,1),this}makeRotation(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,i,n,0,0,0,1),this}makeScale(e,n){return this.set(e,0,0,0,n,0,0,0,1),this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<9;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<9;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}};Ig.prototype.isMatrix3=!0;var Pe=Ig,Um=new Pe,aS=new Pe().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),oS=new Pe().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function nT(){let t={enabled:!0,workingColorSpace:Ts,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===st&&(s.r=Es(s.r),s.g=Es(s.g),s.b=Es(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===st&&(s.r=po(s.r),s.g=po(s.g),s.b=po(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===ni?Pl:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return jr("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),t.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return jr("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),t.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],n=[.2126,.7152,.0722],i=[.3127,.329];return t.define({[Ts]:{primaries:e,whitePoint:i,transfer:Pl,toXYZ:aS,fromXYZ:oS,luminanceCoefficients:n,workingColorSpaceConfig:{unpackColorSpace:Tt},outputColorSpaceConfig:{drawingBufferColorSpace:Tt}},[Tt]:{primaries:e,whitePoint:i,transfer:st,toXYZ:aS,fromXYZ:oS,luminanceCoefficients:n,outputColorSpaceConfig:{drawingBufferColorSpace:Tt}}}),t}var Ye=nT();function Es(t){return t<.04045?t*.0773993808:Math.pow(t*.9478672986+.0521327014,2.4)}function po(t){return t<.0031308?t*12.92:1.055*Math.pow(t,.41666)-.055}var to,Sf=class{static getDataURL(e,n="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{to===void 0&&(to=Nl("canvas")),to.width=e.width,to.height=e.height;let s=to.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=to}return i.toDataURL(n)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let n=Nl("canvas");n.width=e.width,n.height=e.height;let i=n.getContext("2d");i.drawImage(e,0,0,e.width,e.height);let s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Es(r[a]/255)*255;return i.putImageData(s,0,0),n}else if(e.data){let n=e.data.slice(0);for(let i=0;i<n.length;i++)n instanceof Uint8Array||n instanceof Uint8ClampedArray?n[i]=Math.floor(Es(n[i]/255)*255):n[i]=Es(n[i]);return{data:n,width:e.width,height:e.height}}else return Re("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},iT=0,vo=class{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:iT++}),this.uuid=sc(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let n=this.data;return typeof HTMLVideoElement<"u"&&n instanceof HTMLVideoElement?e.set(n.videoWidth,n.videoHeight,0):typeof VideoFrame<"u"&&n instanceof VideoFrame?e.set(n.displayWidth,n.displayHeight,0):n!==null?e.set(n.width,n.height,n.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(Bm(s[a].image)):r.push(Bm(s[a]))}else r=Bm(s);i.url=r}return n||(e.images[this.uuid]=i),i}};function Bm(t){return typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap?Sf.getDataURL(t):t.data?{data:Array.from(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(Re("Texture: Unable to serialize Texture."),{})}var sT=0,Im=new z,Kt=class t extends zn{constructor(e=t.DEFAULT_IMAGE,n=t.DEFAULT_MAPPING,i=Xi,s=Xi,r=xt,a=yr,o=yi,l=jt,c=t.DEFAULT_ANISOTROPY,h=ni){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:sT++}),this.uuid=sc(),this.name="",this.source=new vo(e),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=c,this.format=o,this.internalFormat=null,this.type=l,this.offset=new Ie(0,0),this.repeat=new Ie(1,1),this.center=new Ie(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Pe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.normalized=!1}get width(){return this.source.getSize(Im).x}get height(){return this.source.getSize(Im).y}get depth(){return this.source.getSize(Im).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let n in e){let i=e[n];if(i===void 0){Re(`Texture.setValues(): parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){Re(`Texture.setValues(): property '${n}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),n||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==vg)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case vf:e.x=e.x-Math.floor(e.x);break;case Xi:e.x=e.x<0?0:1;break;case xf:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case vf:e.y=e.y-Math.floor(e.y);break;case Xi:e.y=e.y<0?0:1;break;case xf:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};Kt.DEFAULT_IMAGE=null;Kt.DEFAULT_MAPPING=vg;Kt.DEFAULT_ANISOTROPY=1;var Pg=class Pg{constructor(e=0,n=0,i=0,s=1){this.x=e,this.y=n,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,n,i,s){return this.x=e,this.y=n,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;case 3:this.w=n;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this.w=e.w+n.w,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this.w+=e.w*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this.w=e.w-n.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=this.w,a=e.elements;return this.x=a[0]*n+a[4]*i+a[8]*s+a[12]*r,this.y=a[1]*n+a[5]*i+a[9]*s+a[13]*r,this.z=a[2]*n+a[6]*i+a[10]*s+a[14]*r,this.w=a[3]*n+a[7]*i+a[11]*s+a[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let n=Math.sqrt(1-e.w*e.w);return n<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/n,this.y=e.y/n,this.z=e.z/n),this}setAxisAngleFromRotationMatrix(e){let n,i,s,r,l=e.elements,c=l[0],h=l[4],p=l[8],u=l[1],d=l[5],v=l[9],M=l[2],m=l[6],f=l[10];if(Math.abs(h-u)<.01&&Math.abs(p-M)<.01&&Math.abs(v-m)<.01){if(Math.abs(h+u)<.1&&Math.abs(p+M)<.1&&Math.abs(v+m)<.1&&Math.abs(c+d+f-3)<.1)return this.set(1,0,0,0),this;n=Math.PI;let S=(c+1)/2,_=(d+1)/2,T=(f+1)/2,b=(h+u)/4,w=(p+M)/4,x=(v+m)/4;return S>_&&S>T?S<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(S),s=b/i,r=w/i):_>T?_<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(_),i=b/s,r=x/s):T<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),i=w/r,s=x/r),this.set(i,s,r,n),this}let g=Math.sqrt((m-v)*(m-v)+(p-M)*(p-M)+(u-h)*(u-h));return Math.abs(g)<.001&&(g=1),this.x=(m-v)/g,this.y=(p-M)/g,this.z=(u-h)/g,this.w=Math.acos((c+d+f-1)/2),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this.w=n[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,n){return this.x=Qe(this.x,e.x,n.x),this.y=Qe(this.y,e.y,n.y),this.z=Qe(this.z,e.z,n.z),this.w=Qe(this.w,e.w,n.w),this}clampScalar(e,n){return this.x=Qe(this.x,e,n),this.y=Qe(this.y,e,n),this.z=Qe(this.z,e,n),this.w=Qe(this.w,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar(Qe(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this.w+=(e.w-this.w)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this.w=e.w+(n.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this.w=e[n+3],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e[n+3]=this.w,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this.w=e.getW(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};Pg.prototype.isVector4=!0;var Dt=Pg,Af=class extends zn{constructor(e=1,n=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:xt,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},i),this.isRenderTarget=!0,this.width=e,this.height=n,this.depth=i.depth,this.scissor=new Dt(0,0,e,n),this.scissorTest=!1,this.viewport=new Dt(0,0,e,n),this.textures=[];let s={width:e,height:n,depth:i.depth},r=new Kt(s),a=i.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview,this.useArrayDepthTexture=i.useArrayDepthTexture}_setTextureOptions(e={}){let n={minFilter:xt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(n.mapping=e.mapping),e.wrapS!==void 0&&(n.wrapS=e.wrapS),e.wrapT!==void 0&&(n.wrapT=e.wrapT),e.wrapR!==void 0&&(n.wrapR=e.wrapR),e.magFilter!==void 0&&(n.magFilter=e.magFilter),e.minFilter!==void 0&&(n.minFilter=e.minFilter),e.format!==void 0&&(n.format=e.format),e.type!==void 0&&(n.type=e.type),e.anisotropy!==void 0&&(n.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(n.colorSpace=e.colorSpace),e.flipY!==void 0&&(n.flipY=e.flipY),e.generateMipmaps!==void 0&&(n.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(n.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(n)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,n,i=1){if(this.width!==e||this.height!==n||this.depth!==i){this.width=e,this.height=n,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=n,this.textures[s].image.depth=i,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,n),this.scissor.set(0,0,e,n)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let n=0,i=e.textures.length;n<i;n++){this.textures[n]=e.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0,this.textures[n].renderTarget=this;let s=Object.assign({},e.textures[n].image);this.textures[n].source=new vo(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},Nt=class extends Af{constructor(e=1,n=1,i={}){super(e,n,i),this.isWebGLRenderTarget=!0}},Ol=class extends Kt{constructor(e=null,n=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=un,this.minFilter=un,this.wrapR=Xi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var Mf=class extends Kt{constructor(e=null,n=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=un,this.minFilter=un,this.wrapR=Xi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var Vf=class Vf{constructor(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m)}set(e,n,i,s,r,a,o,l,c,h,p,u,d,v,M,m){let f=this.elements;return f[0]=e,f[4]=n,f[8]=i,f[12]=s,f[1]=r,f[5]=a,f[9]=o,f[13]=l,f[2]=c,f[6]=h,f[10]=p,f[14]=u,f[3]=d,f[7]=v,f[11]=M,f[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Vf().fromArray(this.elements)}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],n[9]=i[9],n[10]=i[10],n[11]=i[11],n[12]=i[12],n[13]=i[13],n[14]=i[14],n[15]=i[15],this}copyPosition(e){let n=this.elements,i=e.elements;return n[12]=i[12],n[13]=i[13],n[14]=i[14],this}setFromMatrix3(e){let n=e.elements;return this.set(n[0],n[3],n[6],0,n[1],n[4],n[7],0,n[2],n[5],n[8],0,0,0,0,1),this}extractBasis(e,n,i){return this.determinantAffine()===0?(e.set(1,0,0),n.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),n.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,n,i){return this.set(e.x,n.x,i.x,0,e.y,n.y,i.y,0,e.z,n.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let n=this.elements,i=e.elements,s=1/no.setFromMatrixColumn(e,0).length(),r=1/no.setFromMatrixColumn(e,1).length(),a=1/no.setFromMatrixColumn(e,2).length();return n[0]=i[0]*s,n[1]=i[1]*s,n[2]=i[2]*s,n[3]=0,n[4]=i[4]*r,n[5]=i[5]*r,n[6]=i[6]*r,n[7]=0,n[8]=i[8]*a,n[9]=i[9]*a,n[10]=i[10]*a,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromEuler(e){let n=this.elements,i=e.x,s=e.y,r=e.z,a=Math.cos(i),o=Math.sin(i),l=Math.cos(s),c=Math.sin(s),h=Math.cos(r),p=Math.sin(r);if(e.order==="XYZ"){let u=a*h,d=a*p,v=o*h,M=o*p;n[0]=l*h,n[4]=-l*p,n[8]=c,n[1]=d+v*c,n[5]=u-M*c,n[9]=-o*l,n[2]=M-u*c,n[6]=v+d*c,n[10]=a*l}else if(e.order==="YXZ"){let u=l*h,d=l*p,v=c*h,M=c*p;n[0]=u+M*o,n[4]=v*o-d,n[8]=a*c,n[1]=a*p,n[5]=a*h,n[9]=-o,n[2]=d*o-v,n[6]=M+u*o,n[10]=a*l}else if(e.order==="ZXY"){let u=l*h,d=l*p,v=c*h,M=c*p;n[0]=u-M*o,n[4]=-a*p,n[8]=v+d*o,n[1]=d+v*o,n[5]=a*h,n[9]=M-u*o,n[2]=-a*c,n[6]=o,n[10]=a*l}else if(e.order==="ZYX"){let u=a*h,d=a*p,v=o*h,M=o*p;n[0]=l*h,n[4]=v*c-d,n[8]=u*c+M,n[1]=l*p,n[5]=M*c+u,n[9]=d*c-v,n[2]=-c,n[6]=o*l,n[10]=a*l}else if(e.order==="YZX"){let u=a*l,d=a*c,v=o*l,M=o*c;n[0]=l*h,n[4]=M-u*p,n[8]=v*p+d,n[1]=p,n[5]=a*h,n[9]=-o*h,n[2]=-c*h,n[6]=d*p+v,n[10]=u-M*p}else if(e.order==="XZY"){let u=a*l,d=a*c,v=o*l,M=o*c;n[0]=l*h,n[4]=-p,n[8]=c*h,n[1]=u*p+M,n[5]=a*h,n[9]=d*p-v,n[2]=v*p-d,n[6]=o*h,n[10]=M*p+u}return n[3]=0,n[7]=0,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromQuaternion(e){return this.compose(rT,e,aT)}lookAt(e,n,i){let s=this.elements;return jn.subVectors(e,n),jn.lengthSq()===0&&(jn.z=1),jn.normalize(),sr.crossVectors(i,jn),sr.lengthSq()===0&&(Math.abs(i.z)===1?jn.x+=1e-4:jn.z+=1e-4,jn.normalize(),sr.crossVectors(i,jn)),sr.normalize(),Hu.crossVectors(jn,sr),s[0]=sr.x,s[4]=Hu.x,s[8]=jn.x,s[1]=sr.y,s[5]=Hu.y,s[9]=jn.y,s[2]=sr.z,s[6]=Hu.z,s[10]=jn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[4],l=i[8],c=i[12],h=i[1],p=i[5],u=i[9],d=i[13],v=i[2],M=i[6],m=i[10],f=i[14],g=i[3],S=i[7],_=i[11],T=i[15],b=s[0],w=s[4],x=s[8],E=s[12],R=s[1],D=s[5],L=s[9],q=s[13],Y=s[2],N=s[6],k=s[10],V=s[14],j=s[3],ee=s[7],se=s[11],he=s[15];return r[0]=a*b+o*R+l*Y+c*j,r[4]=a*w+o*D+l*N+c*ee,r[8]=a*x+o*L+l*k+c*se,r[12]=a*E+o*q+l*V+c*he,r[1]=h*b+p*R+u*Y+d*j,r[5]=h*w+p*D+u*N+d*ee,r[9]=h*x+p*L+u*k+d*se,r[13]=h*E+p*q+u*V+d*he,r[2]=v*b+M*R+m*Y+f*j,r[6]=v*w+M*D+m*N+f*ee,r[10]=v*x+M*L+m*k+f*se,r[14]=v*E+M*q+m*V+f*he,r[3]=g*b+S*R+_*Y+T*j,r[7]=g*w+S*D+_*N+T*ee,r[11]=g*x+S*L+_*k+T*se,r[15]=g*E+S*q+_*V+T*he,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[4]*=e,n[8]*=e,n[12]*=e,n[1]*=e,n[5]*=e,n[9]*=e,n[13]*=e,n[2]*=e,n[6]*=e,n[10]*=e,n[14]*=e,n[3]*=e,n[7]*=e,n[11]*=e,n[15]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[12],a=e[1],o=e[5],l=e[9],c=e[13],h=e[2],p=e[6],u=e[10],d=e[14],v=e[3],M=e[7],m=e[11],f=e[15],g=l*d-c*u,S=o*d-c*p,_=o*u-l*p,T=a*d-c*h,b=a*u-l*h,w=a*p-o*h;return n*(M*g-m*S+f*_)-i*(v*g-m*T+f*b)+s*(v*S-M*T+f*w)-r*(v*_-M*b+m*w)}determinantAffine(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[1],a=e[5],o=e[9],l=e[2],c=e[6],h=e[10];return n*(a*h-o*c)-i*(r*h-o*l)+s*(r*c-a*l)}transpose(){let e=this.elements,n;return n=e[1],e[1]=e[4],e[4]=n,n=e[2],e[2]=e[8],e[8]=n,n=e[6],e[6]=e[9],e[9]=n,n=e[3],e[3]=e[12],e[12]=n,n=e[7],e[7]=e[13],e[13]=n,n=e[11],e[11]=e[14],e[14]=n,this}setPosition(e,n,i){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=n,s[14]=i),this}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=e[9],u=e[10],d=e[11],v=e[12],M=e[13],m=e[14],f=e[15],g=n*o-i*a,S=n*l-s*a,_=n*c-r*a,T=i*l-s*o,b=i*c-r*o,w=s*c-r*l,x=h*M-p*v,E=h*m-u*v,R=h*f-d*v,D=p*m-u*M,L=p*f-d*M,q=u*f-d*m,Y=g*q-S*L+_*D+T*R-b*E+w*x;if(Y===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let N=1/Y;return e[0]=(o*q-l*L+c*D)*N,e[1]=(s*L-i*q-r*D)*N,e[2]=(M*w-m*b+f*T)*N,e[3]=(u*b-p*w-d*T)*N,e[4]=(l*R-a*q-c*E)*N,e[5]=(n*q-s*R+r*E)*N,e[6]=(m*_-v*w-f*S)*N,e[7]=(h*w-u*_+d*S)*N,e[8]=(a*L-o*R+c*x)*N,e[9]=(i*R-n*L-r*x)*N,e[10]=(v*b-M*_+f*g)*N,e[11]=(p*_-h*b-d*g)*N,e[12]=(o*E-a*D-l*x)*N,e[13]=(n*D-i*E+s*x)*N,e[14]=(M*S-v*T-m*g)*N,e[15]=(h*T-p*S+u*g)*N,this}scale(e){let n=this.elements,i=e.x,s=e.y,r=e.z;return n[0]*=i,n[4]*=s,n[8]*=r,n[1]*=i,n[5]*=s,n[9]*=r,n[2]*=i,n[6]*=s,n[10]*=r,n[3]*=i,n[7]*=s,n[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,n=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(n,i,s))}makeTranslation(e,n,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,n,0,0,1,i,0,0,0,1),this}makeRotationX(e){let n=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,n,-i,0,0,i,n,0,0,0,0,1),this}makeRotationY(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,0,i,0,0,1,0,0,-i,0,n,0,0,0,0,1),this}makeRotationZ(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,0,i,n,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,n){let i=Math.cos(n),s=Math.sin(n),r=1-i,a=e.x,o=e.y,l=e.z,c=r*a,h=r*o;return this.set(c*a+i,c*o-s*l,c*l+s*o,0,c*o+s*l,h*o+i,h*l-s*a,0,c*l-s*o,h*l+s*a,r*l*l+i,0,0,0,0,1),this}makeScale(e,n,i){return this.set(e,0,0,0,0,n,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,n,i,s,r,a){return this.set(1,i,r,0,e,1,a,0,n,s,1,0,0,0,0,1),this}compose(e,n,i){let s=this.elements,r=n._x,a=n._y,o=n._z,l=n._w,c=r+r,h=a+a,p=o+o,u=r*c,d=r*h,v=r*p,M=a*h,m=a*p,f=o*p,g=l*c,S=l*h,_=l*p,T=i.x,b=i.y,w=i.z;return s[0]=(1-(M+f))*T,s[1]=(d+_)*T,s[2]=(v-S)*T,s[3]=0,s[4]=(d-_)*b,s[5]=(1-(u+f))*b,s[6]=(m+g)*b,s[7]=0,s[8]=(v+S)*w,s[9]=(m-g)*w,s[10]=(1-(u+M))*w,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,n,i){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return i.set(1,1,1),n.identity(),this;let a=no.set(s[0],s[1],s[2]).length(),o=no.set(s[4],s[5],s[6]).length(),l=no.set(s[8],s[9],s[10]).length();r<0&&(a=-a),Mi.copy(this);let c=1/a,h=1/o,p=1/l;return Mi.elements[0]*=c,Mi.elements[1]*=c,Mi.elements[2]*=c,Mi.elements[4]*=h,Mi.elements[5]*=h,Mi.elements[6]*=h,Mi.elements[8]*=p,Mi.elements[9]*=p,Mi.elements[10]*=p,n.setFromRotationMatrix(Mi),i.x=a,i.y=o,i.z=l,this}makePerspective(e,n,i,s,r,a,o=bi,l=!1){let c=this.elements,h=2*r/(n-e),p=2*r/(i-s),u=(n+e)/(n-e),d=(i+s)/(i-s),v,M;if(l)v=r/(a-r),M=a*r/(a-r);else if(o===bi)v=-(a+r)/(a-r),M=-2*a*r/(a-r);else if(o===Ll)v=-a/(a-r),M=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=u,c[12]=0,c[1]=0,c[5]=p,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=v,c[14]=M,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,n,i,s,r,a,o=bi,l=!1){let c=this.elements,h=2/(n-e),p=2/(i-s),u=-(n+e)/(n-e),d=-(i+s)/(i-s),v,M;if(l)v=1/(a-r),M=a/(a-r);else if(o===bi)v=-2/(a-r),M=-(a+r)/(a-r);else if(o===Ll)v=-1/(a-r),M=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=0,c[12]=u,c[1]=0,c[5]=p,c[9]=0,c[13]=d,c[2]=0,c[6]=0,c[10]=v,c[14]=M,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<16;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<16;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e[n+9]=i[9],e[n+10]=i[10],e[n+11]=i[11],e[n+12]=i[12],e[n+13]=i[13],e[n+14]=i[14],e[n+15]=i[15],e}};Vf.prototype.isMatrix4=!0;var Ht=Vf,no=new z,Mi=new Ht,rT=new z(0,0,0),aT=new z(1,1,1),sr=new z,Hu=new z,jn=new z,lS=new Ht,cS=new qi,hr=class t{constructor(e=0,n=0,i=0,s=t.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=n,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,n,i,s=this._order){return this._x=e,this._y=n,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,n=this._order,i=!0){let s=e.elements,r=s[0],a=s[4],o=s[8],l=s[1],c=s[5],h=s[9],p=s[2],u=s[6],d=s[10];switch(n){case"XYZ":this._y=Math.asin(Qe(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,d),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(u,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Qe(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,d),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-p,r),this._z=0);break;case"ZXY":this._x=Math.asin(Qe(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-p,d),this._z=Math.atan2(-a,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Qe(p,-1,1)),Math.abs(p)<.9999999?(this._x=Math.atan2(u,d),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-a,c));break;case"YZX":this._z=Math.asin(Qe(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-h,c),this._y=Math.atan2(-p,r)):(this._x=0,this._y=Math.atan2(o,d));break;case"XZY":this._z=Math.asin(-Qe(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(u,c),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,d),this._y=0);break;default:Re("Euler: .setFromRotationMatrix() encountered an unknown order: "+n)}return this._order=n,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,n,i){return lS.makeRotationFromQuaternion(e),this.setFromRotationMatrix(lS,n,i)}setFromVector3(e,n=this._order){return this.set(e.x,e.y,e.z,n)}reorder(e){return cS.setFromEuler(this),this.setFromQuaternion(cS,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};hr.DEFAULT_ORDER="XYZ";var Fl=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},oT=0,uS=new z,io=new qi,ys=new Ht,Vu=new z,Rl=new z,lT=new z,cT=new qi,fS=new z(1,0,0),hS=new z(0,1,0),dS=new z(0,0,1),pS={type:"added"},uT={type:"removed"},so={type:"childadded",child:null},Pm={type:"childremoved",child:null},vi=class t extends zn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:oT++}),this.uuid=sc(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=t.DEFAULT_UP.clone();let e=new z,n=new hr,i=new qi,s=new z(1,1,1);function r(){i.setFromEuler(n,!1)}function a(){n.setFromQuaternion(i,void 0,!1)}n._onChange(r),i._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new Ht},normalMatrix:{value:new Pe}}),this.matrix=new Ht,this.matrixWorld=new Ht,this.matrixAutoUpdate=t.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=t.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Fl,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,n){this.quaternion.setFromAxisAngle(e,n)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,n){return io.setFromAxisAngle(e,n),this.quaternion.multiply(io),this}rotateOnWorldAxis(e,n){return io.setFromAxisAngle(e,n),this.quaternion.premultiply(io),this}rotateX(e){return this.rotateOnAxis(fS,e)}rotateY(e){return this.rotateOnAxis(hS,e)}rotateZ(e){return this.rotateOnAxis(dS,e)}translateOnAxis(e,n){return uS.copy(e).applyQuaternion(this.quaternion),this.position.add(uS.multiplyScalar(n)),this}translateX(e){return this.translateOnAxis(fS,e)}translateY(e){return this.translateOnAxis(hS,e)}translateZ(e){return this.translateOnAxis(dS,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(ys.copy(this.matrixWorld).invert())}lookAt(e,n,i){e.isVector3?Vu.copy(e):Vu.set(e,n,i);let s=this.parent;this.updateWorldMatrix(!0,!1),Rl.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?ys.lookAt(Rl,Vu,this.up):ys.lookAt(Vu,Rl,this.up),this.quaternion.setFromRotationMatrix(ys),s&&(ys.extractRotation(s.matrixWorld),io.setFromRotationMatrix(ys),this.quaternion.premultiply(io.invert()))}add(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.add(arguments[n]);return this}return e===this?(Ue("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(pS),so.child=e,this.dispatchEvent(so),so.child=null):Ue("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let n=this.children.indexOf(e);return n!==-1&&(e.parent=null,this.children.splice(n,1),e.dispatchEvent(uT),Pm.child=e,this.dispatchEvent(Pm),Pm.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),ys.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),ys.multiply(e.parent.matrixWorld)),e.applyMatrix4(ys),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(pS),so.child=e,this.dispatchEvent(so),so.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,n){if(this[e]===n)return this;for(let i=0,s=this.children.length;i<s;i++){let a=this.children[i].getObjectByProperty(e,n);if(a!==void 0)return a}}getObjectsByProperty(e,n,i=[]){this[e]===n&&i.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(e,n,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Rl,e,lT),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Rl,cT,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let n=this.matrixWorld.elements;return e.set(n[8],n[9],n[10]).normalize()}raycast(){}traverse(e){e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverseVisible(e)}traverseAncestors(e){let n=this.parent;n!==null&&(e(n),n.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let n=e.x,i=e.y,s=e.z,r=this.matrix.elements;r[12]+=n-r[0]*n-r[4]*i-r[8]*s,r[13]+=i-r[1]*n-r[5]*i-r[9]*s,r[14]+=s-r[2]*n-r[6]*i-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].updateMatrixWorld(e)}updateWorldMatrix(e,n,i=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||i)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,i=!0),n===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,i)}}toJSON(e){let n=e===void 0||typeof e=="string",i={};n&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),this.static!==!1&&(s.static=this.static),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,l){return o[l.uuid]===void 0&&(o[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let l=o.shapes;if(Array.isArray(l))for(let c=0,h=l.length;c<h;c++){let p=l[c];r(e.shapes,p)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let l=0,c=this.material.length;l<c;l++)o.push(r(e.materials,this.material[l]));s.material=o}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let l=this.animations[o];s.animations.push(r(e.animations,l))}}if(n){let o=a(e.geometries),l=a(e.materials),c=a(e.textures),h=a(e.images),p=a(e.shapes),u=a(e.skeletons),d=a(e.animations),v=a(e.nodes);o.length>0&&(i.geometries=o),l.length>0&&(i.materials=l),c.length>0&&(i.textures=c),h.length>0&&(i.images=h),p.length>0&&(i.shapes=p),u.length>0&&(i.skeletons=u),d.length>0&&(i.animations=d),v.length>0&&(i.nodes=v)}return i.object=s,i;function a(o){let l=[];for(let c in o){let h=o[c];delete h.metadata,l.push(h)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,n=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),n===!0)for(let i=0;i<e.children.length;i++){let s=e.children[i];this.add(s.clone())}return this}};vi.DEFAULT_UP=new z(0,1,0);vi.DEFAULT_MATRIX_AUTO_UPDATE=!0;vi.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var Jr=class extends vi{constructor(){super(),this.isGroup=!0,this.type="Group"}},fT={type:"move"},xo=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Jr,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Jr,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new z,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new z),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Jr,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new z,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new z,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let n=this._hand;if(n)for(let i of e.hand.values())this._getHandJoint(n,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,n,i){let s=null,r=null,a=null,o=this._targetRay,l=this._grip,c=this._hand;if(e&&n.session.visibilityState!=="visible-blurred"){if(c&&e.hand){a=!0;for(let M of e.hand.values()){let m=n.getJointPose(M,i),f=this._getHandJoint(c,M);m!==null&&(f.matrix.fromArray(m.transform.matrix),f.matrix.decompose(f.position,f.rotation,f.scale),f.matrixWorldNeedsUpdate=!0,f.jointRadius=m.radius),f.visible=m!==null}let h=c.joints["index-finger-tip"],p=c.joints["thumb-tip"],u=h.position.distanceTo(p.position),d=.02,v=.005;c.inputState.pinching&&u>d+v?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&u<=d-v&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=n.getPose(e.gripSpace,i),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1,l.eventsEnabled&&l.dispatchEvent({type:"gripUpdated",data:e,target:this})));o!==null&&(s=n.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(fT)))}return o!==null&&(o.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=a!==null),this}_getHandJoint(e,n){if(e.joints[n.jointName]===void 0){let i=new Jr;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[n.jointName]=i,e.add(i)}return e.joints[n.jointName]}},iA={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},rr={h:0,s:0,l:0},ku={h:0,s:0,l:0};function Lm(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<1/6?t+(e-t)*6*n:n<1/2?e:n<2/3?t+(e-t)*6*(2/3-n):t}var ke=class{constructor(e,n,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,n,i)}set(e,n,i){if(n===void 0&&i===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,n,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,n=Tt){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,Ye.colorSpaceToWorking(this,n),this}setRGB(e,n,i,s=Ye.workingColorSpace){return this.r=e,this.g=n,this.b=i,Ye.colorSpaceToWorking(this,s),this}setHSL(e,n,i,s=Ye.workingColorSpace){if(e=tT(e,1),n=Qe(n,0,1),i=Qe(i,0,1),n===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+n):i+n-i*n,a=2*i-r;this.r=Lm(a,r,e+1/3),this.g=Lm(a,r,e),this.b=Lm(a,r,e-1/3)}return Ye.colorSpaceToWorking(this,s),this}setStyle(e,n=Tt){function i(r){r!==void 0&&parseFloat(r)<1&&Re("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,n);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,n);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,n);break;default:Re("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,n);if(a===6)return this.setHex(parseInt(r,16),n);Re("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,n);return this}setColorName(e,n=Tt){let i=iA[e.toLowerCase()];return i!==void 0?this.setHex(i,n):Re("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Es(e.r),this.g=Es(e.g),this.b=Es(e.b),this}copyLinearToSRGB(e){return this.r=po(e.r),this.g=po(e.g),this.b=po(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Tt){return Ye.workingToColorSpace(_n.copy(this),e),Math.round(Qe(_n.r*255,0,255))*65536+Math.round(Qe(_n.g*255,0,255))*256+Math.round(Qe(_n.b*255,0,255))}getHexString(e=Tt){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,n=Ye.workingColorSpace){Ye.workingToColorSpace(_n.copy(this),n);let i=_n.r,s=_n.g,r=_n.b,a=Math.max(i,s,r),o=Math.min(i,s,r),l,c,h=(o+a)/2;if(o===a)l=0,c=0;else{let p=a-o;switch(c=h<=.5?p/(a+o):p/(2-a-o),a){case i:l=(s-r)/p+(s<r?6:0);break;case s:l=(r-i)/p+2;break;case r:l=(i-s)/p+4;break}l/=6}return e.h=l,e.s=c,e.l=h,e}getRGB(e,n=Ye.workingColorSpace){return Ye.workingToColorSpace(_n.copy(this),n),e.r=_n.r,e.g=_n.g,e.b=_n.b,e}getStyle(e=Tt){Ye.workingToColorSpace(_n.copy(this),e);let n=_n.r,i=_n.g,s=_n.b;return e!==Tt?`color(${e} ${n.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(n*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,n,i){return this.getHSL(rr),this.setHSL(rr.h+e,rr.s+n,rr.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,n){return this.r=e.r+n.r,this.g=e.g+n.g,this.b=e.b+n.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,n){return this.r+=(e.r-this.r)*n,this.g+=(e.g-this.g)*n,this.b+=(e.b-this.b)*n,this}lerpColors(e,n,i){return this.r=e.r+(n.r-e.r)*i,this.g=e.g+(n.g-e.g)*i,this.b=e.b+(n.b-e.b)*i,this}lerpHSL(e,n){this.getHSL(rr),e.getHSL(ku);let i=Rm(rr.h,ku.h,n),s=Rm(rr.s,ku.s,n),r=Rm(rr.l,ku.l,n);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let n=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*n+r[3]*i+r[6]*s,this.g=r[1]*n+r[4]*i+r[7]*s,this.b=r[2]*n+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,n=0){return this.r=e[n],this.g=e[n+1],this.b=e[n+2],this}toArray(e=[],n=0){return e[n]=this.r,e[n+1]=this.g,e[n+2]=this.b,e}fromBufferAttribute(e,n){return this.r=e.getX(n),this.g=e.getY(n),this.b=e.getZ(n),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},_n=new ke;ke.NAMES=iA;var dr=class extends vi{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new hr,this.environmentIntensity=1,this.environmentRotation=new hr,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,n){return super.copy(e,n),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let n=super.toJSON(e);return this.fog!==null&&(n.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(n.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(n.object.backgroundIntensity=this.backgroundIntensity),n.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(n.object.environmentIntensity=this.environmentIntensity),n.object.environmentRotation=this.environmentRotation.toArray(),n}},Ei=new z,_s=new z,Nm=new z,Ss=new z,ro=new z,ao=new z,mS=new z,Om=new z,Fm=new z,zm=new z,Gm=new Dt,Hm=new Dt,Vm=new Dt,ur=class t{constructor(e=new z,n=new z,i=new z){this.a=e,this.b=n,this.c=i}static getNormal(e,n,i,s){s.subVectors(i,n),Ei.subVectors(e,n),s.cross(Ei);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,n,i,s,r){Ei.subVectors(s,n),_s.subVectors(i,n),Nm.subVectors(e,n);let a=Ei.dot(Ei),o=Ei.dot(_s),l=Ei.dot(Nm),c=_s.dot(_s),h=_s.dot(Nm),p=a*c-o*o;if(p===0)return r.set(0,0,0),null;let u=1/p,d=(c*l-o*h)*u,v=(a*h-o*l)*u;return r.set(1-d-v,v,d)}static containsPoint(e,n,i,s){return this.getBarycoord(e,n,i,s,Ss)===null?!1:Ss.x>=0&&Ss.y>=0&&Ss.x+Ss.y<=1}static getInterpolation(e,n,i,s,r,a,o,l){return this.getBarycoord(e,n,i,s,Ss)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Ss.x),l.addScaledVector(a,Ss.y),l.addScaledVector(o,Ss.z),l)}static getInterpolatedAttribute(e,n,i,s,r,a){return Gm.setScalar(0),Hm.setScalar(0),Vm.setScalar(0),Gm.fromBufferAttribute(e,n),Hm.fromBufferAttribute(e,i),Vm.fromBufferAttribute(e,s),a.setScalar(0),a.addScaledVector(Gm,r.x),a.addScaledVector(Hm,r.y),a.addScaledVector(Vm,r.z),a}static isFrontFacing(e,n,i,s){return Ei.subVectors(i,n),_s.subVectors(e,n),Ei.cross(_s).dot(s)<0}set(e,n,i){return this.a.copy(e),this.b.copy(n),this.c.copy(i),this}setFromPointsAndIndices(e,n,i,s){return this.a.copy(e[n]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,n,i,s){return this.a.fromBufferAttribute(e,n),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Ei.subVectors(this.c,this.b),_s.subVectors(this.a,this.b),Ei.cross(_s).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return t.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,n){return t.getBarycoord(e,this.a,this.b,this.c,n)}getInterpolation(e,n,i,s,r){return t.getInterpolation(e,this.a,this.b,this.c,n,i,s,r)}containsPoint(e){return t.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return t.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,n){let i=this.a,s=this.b,r=this.c,a,o;ro.subVectors(s,i),ao.subVectors(r,i),Om.subVectors(e,i);let l=ro.dot(Om),c=ao.dot(Om);if(l<=0&&c<=0)return n.copy(i);Fm.subVectors(e,s);let h=ro.dot(Fm),p=ao.dot(Fm);if(h>=0&&p<=h)return n.copy(s);let u=l*p-h*c;if(u<=0&&l>=0&&h<=0)return a=l/(l-h),n.copy(i).addScaledVector(ro,a);zm.subVectors(e,r);let d=ro.dot(zm),v=ao.dot(zm);if(v>=0&&d<=v)return n.copy(r);let M=d*c-l*v;if(M<=0&&c>=0&&v<=0)return o=c/(c-v),n.copy(i).addScaledVector(ao,o);let m=h*v-d*p;if(m<=0&&p-h>=0&&d-v>=0)return mS.subVectors(r,s),o=(p-h)/(p-h+(d-v)),n.copy(s).addScaledVector(mS,o);let f=1/(m+M+u);return a=M*f,o=u*f,n.copy(i).addScaledVector(ro,a).addScaledVector(ao,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},pr=class{constructor(e=new z(1/0,1/0,1/0),n=new z(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=n}set(e,n){return this.min.copy(e),this.max.copy(n),this}setFromArray(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n+=3)this.expandByPoint(Ti.fromArray(e,n));return this}setFromBufferAttribute(e){this.makeEmpty();for(let n=0,i=e.count;n<i;n++)this.expandByPoint(Ti.fromBufferAttribute(e,n));return this}setFromPoints(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n++)this.expandByPoint(e[n]);return this}setFromCenterAndSize(e,n){let i=Ti.copy(n).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,n=!1){return this.makeEmpty(),this.expandByObject(e,n)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,n=!1){e.updateWorldMatrix(!1,!1);let i=e.geometry;if(i!==void 0){let r=i.getAttribute("position");if(n===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,Ti):Ti.fromBufferAttribute(r,a),Ti.applyMatrix4(e.matrixWorld),this.expandByPoint(Ti);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Wu.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Wu.copy(i.boundingBox)),Wu.applyMatrix4(e.matrixWorld),this.union(Wu)}let s=e.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],n);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,n){return n.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Ti),Ti.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let n,i;return e.normal.x>0?(n=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(n=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(n+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(n+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(n+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(n+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),n<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(Dl),Xu.subVectors(this.max,Dl),oo.subVectors(e.a,Dl),lo.subVectors(e.b,Dl),co.subVectors(e.c,Dl),ar.subVectors(lo,oo),or.subVectors(co,lo),Yr.subVectors(oo,co);let n=[0,-ar.z,ar.y,0,-or.z,or.y,0,-Yr.z,Yr.y,ar.z,0,-ar.x,or.z,0,-or.x,Yr.z,0,-Yr.x,-ar.y,ar.x,0,-or.y,or.x,0,-Yr.y,Yr.x,0];return!km(n,oo,lo,co,Xu)||(n=[1,0,0,0,1,0,0,0,1],!km(n,oo,lo,co,Xu))?!1:(Yu.crossVectors(ar,or),n=[Yu.x,Yu.y,Yu.z],km(n,oo,lo,co,Xu))}clampPoint(e,n){return n.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Ti).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Ti).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(As[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),As[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),As[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),As[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),As[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),As[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),As[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),As[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(As),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},As=[new z,new z,new z,new z,new z,new z,new z,new z],Ti=new z,Wu=new pr,oo=new z,lo=new z,co=new z,ar=new z,or=new z,Yr=new z,Dl=new z,Xu=new z,Yu=new z,qr=new z;function km(t,e,n,i,s){for(let r=0,a=t.length-3;r<=a;r+=3){qr.fromArray(t,r);let o=s.x*Math.abs(qr.x)+s.y*Math.abs(qr.y)+s.z*Math.abs(qr.z),l=e.dot(qr),c=n.dot(qr),h=i.dot(qr);if(Math.max(-Math.max(l,c,h),Math.min(l,c,h))>o)return!1}return!0}var Zt=new z,qu=new Ie,hT=0,An=class extends zn{constructor(e,n,i=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:hT++}),this.name="",this.array=e,this.itemSize=n,this.count=e!==void 0?e.length/n:0,this.normalized=i,this.usage=ng,this.updateRanges=[],this.gpuType=ti,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,n,i){e*=this.itemSize,i*=n.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=n.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let n=0,i=this.count;n<i;n++)qu.fromBufferAttribute(this,n),qu.applyMatrix3(e),this.setXY(n,qu.x,qu.y);else if(this.itemSize===3)for(let n=0,i=this.count;n<i;n++)Zt.fromBufferAttribute(this,n),Zt.applyMatrix3(e),this.setXYZ(n,Zt.x,Zt.y,Zt.z);return this}applyMatrix4(e){for(let n=0,i=this.count;n<i;n++)Zt.fromBufferAttribute(this,n),Zt.applyMatrix4(e),this.setXYZ(n,Zt.x,Zt.y,Zt.z);return this}applyNormalMatrix(e){for(let n=0,i=this.count;n<i;n++)Zt.fromBufferAttribute(this,n),Zt.applyNormalMatrix(e),this.setXYZ(n,Zt.x,Zt.y,Zt.z);return this}transformDirection(e){for(let n=0,i=this.count;n<i;n++)Zt.fromBufferAttribute(this,n),Zt.transformDirection(e),this.setXYZ(n,Zt.x,Zt.y,Zt.z);return this}set(e,n=0){return this.array.set(e,n),this}getComponent(e,n){let i=this.array[e*this.itemSize+n];return this.normalized&&(i=Cl(i,this.array)),i}setComponent(e,n,i){return this.normalized&&(i=Fn(i,this.array)),this.array[e*this.itemSize+n]=i,this}getX(e){let n=this.array[e*this.itemSize];return this.normalized&&(n=Cl(n,this.array)),n}setX(e,n){return this.normalized&&(n=Fn(n,this.array)),this.array[e*this.itemSize]=n,this}getY(e){let n=this.array[e*this.itemSize+1];return this.normalized&&(n=Cl(n,this.array)),n}setY(e,n){return this.normalized&&(n=Fn(n,this.array)),this.array[e*this.itemSize+1]=n,this}getZ(e){let n=this.array[e*this.itemSize+2];return this.normalized&&(n=Cl(n,this.array)),n}setZ(e,n){return this.normalized&&(n=Fn(n,this.array)),this.array[e*this.itemSize+2]=n,this}getW(e){let n=this.array[e*this.itemSize+3];return this.normalized&&(n=Cl(n,this.array)),n}setW(e,n){return this.normalized&&(n=Fn(n,this.array)),this.array[e*this.itemSize+3]=n,this}setXY(e,n,i){return e*=this.itemSize,this.normalized&&(n=Fn(n,this.array),i=Fn(i,this.array)),this.array[e+0]=n,this.array[e+1]=i,this}setXYZ(e,n,i,s){return e*=this.itemSize,this.normalized&&(n=Fn(n,this.array),i=Fn(i,this.array),s=Fn(s,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,n,i,s,r){return e*=this.itemSize,this.normalized&&(n=Fn(n,this.array),i=Fn(i,this.array),s=Fn(s,this.array),r=Fn(r,this.array)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==ng&&(e.usage=this.usage),e}dispose(){this.dispatchEvent({type:"dispose"})}};var zl=class extends An{constructor(e,n,i){super(new Uint16Array(e),n,i)}};var Gl=class extends An{constructor(e,n,i){super(new Uint32Array(e),n,i)}};var gi=class extends An{constructor(e,n,i){super(new Float32Array(e),n,i)}},dT=new pr,Ul=new z,Wm=new z,yo=class{constructor(e=new z,n=-1){this.isSphere=!0,this.center=e,this.radius=n}set(e,n){return this.center.copy(e),this.radius=n,this}setFromPoints(e,n){let i=this.center;n!==void 0?i.copy(n):dT.setFromPoints(e).getCenter(i);let s=0;for(let r=0,a=e.length;r<a;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let n=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=n*n}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,n){let i=this.center.distanceToSquared(e);return n.copy(e),i>this.radius*this.radius&&(n.sub(this.center).normalize(),n.multiplyScalar(this.radius).add(this.center)),n}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Ul.subVectors(e,this.center);let n=Ul.lengthSq();if(n>this.radius*this.radius){let i=Math.sqrt(n),s=(i-this.radius)*.5;this.center.addScaledVector(Ul,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Wm.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Ul.copy(e.center).add(Wm)),this.expandByPoint(Ul.copy(e.center).sub(Wm))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},pT=0,mi=new Ht,Xm=new vi,uo=new z,$n=new pr,Bl=new pr,cn=new z,xi=class t extends zn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:pT++}),this.uuid=sc(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new($E(e)?Gl:zl)(e,1):this.index=e,this}setIndirect(e,n=0){return this.indirect=e,this.indirectOffset=n,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,n){return this.attributes[e]=n,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,n,i=0){this.groups.push({start:e,count:n,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,n){this.drawRange.start=e,this.drawRange.count=n}applyMatrix4(e){let n=this.attributes.position;n!==void 0&&(n.applyMatrix4(e),n.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new Pe().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return mi.makeRotationFromQuaternion(e),this.applyMatrix4(mi),this}rotateX(e){return mi.makeRotationX(e),this.applyMatrix4(mi),this}rotateY(e){return mi.makeRotationY(e),this.applyMatrix4(mi),this}rotateZ(e){return mi.makeRotationZ(e),this.applyMatrix4(mi),this}translate(e,n,i){return mi.makeTranslation(e,n,i),this.applyMatrix4(mi),this}scale(e,n,i){return mi.makeScale(e,n,i),this.applyMatrix4(mi),this}lookAt(e){return Xm.lookAt(e),Xm.updateMatrix(),this.applyMatrix4(Xm.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(uo).negate(),this.translate(uo.x,uo.y,uo.z),this}setFromPoints(e){let n=this.getAttribute("position");if(n===void 0){let i=[];for(let s=0,r=e.length;s<r;s++){let a=e[s];i.push(a.x,a.y,a.z||0)}this.setAttribute("position",new gi(i,3))}else{let i=Math.min(e.length,n.count);for(let s=0;s<i;s++){let r=e[s];n.setXYZ(s,r.x,r.y,r.z||0)}e.length>n.count&&Re("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),n.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new pr);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ue("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new z(-1/0,-1/0,-1/0),new z(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),n)for(let i=0,s=n.length;i<s;i++){let r=n[i];$n.setFromBufferAttribute(r),this.morphTargetsRelative?(cn.addVectors(this.boundingBox.min,$n.min),this.boundingBox.expandByPoint(cn),cn.addVectors(this.boundingBox.max,$n.max),this.boundingBox.expandByPoint(cn)):(this.boundingBox.expandByPoint($n.min),this.boundingBox.expandByPoint($n.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Ue('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new yo);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Ue("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new z,1/0);return}if(e){let i=this.boundingSphere.center;if($n.setFromBufferAttribute(e),n)for(let r=0,a=n.length;r<a;r++){let o=n[r];Bl.setFromBufferAttribute(o),this.morphTargetsRelative?(cn.addVectors($n.min,Bl.min),$n.expandByPoint(cn),cn.addVectors($n.max,Bl.max),$n.expandByPoint(cn)):($n.expandByPoint(Bl.min),$n.expandByPoint(Bl.max))}$n.getCenter(i);let s=0;for(let r=0,a=e.count;r<a;r++)cn.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(cn));if(n)for(let r=0,a=n.length;r<a;r++){let o=n[r],l=this.morphTargetsRelative;for(let c=0,h=o.count;c<h;c++)cn.fromBufferAttribute(o,c),l&&(uo.fromBufferAttribute(e,c),cn.add(uo)),s=Math.max(s,i.distanceToSquared(cn))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Ue('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,n=this.attributes;if(e===null||n.position===void 0||n.normal===void 0||n.uv===void 0){Ue("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=n.position,s=n.normal,r=n.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==i.count)&&(a=new An(new Float32Array(4*i.count),4),this.setAttribute("tangent",a));let o=[],l=[];for(let x=0;x<i.count;x++)o[x]=new z,l[x]=new z;let c=new z,h=new z,p=new z,u=new Ie,d=new Ie,v=new Ie,M=new z,m=new z;function f(x,E,R){c.fromBufferAttribute(i,x),h.fromBufferAttribute(i,E),p.fromBufferAttribute(i,R),u.fromBufferAttribute(r,x),d.fromBufferAttribute(r,E),v.fromBufferAttribute(r,R),h.sub(c),p.sub(c),d.sub(u),v.sub(u);let D=1/(d.x*v.y-v.x*d.y);isFinite(D)&&(M.copy(h).multiplyScalar(v.y).addScaledVector(p,-d.y).multiplyScalar(D),m.copy(p).multiplyScalar(d.x).addScaledVector(h,-v.x).multiplyScalar(D),o[x].add(M),o[E].add(M),o[R].add(M),l[x].add(m),l[E].add(m),l[R].add(m))}let g=this.groups;g.length===0&&(g=[{start:0,count:e.count}]);for(let x=0,E=g.length;x<E;++x){let R=g[x],D=R.start,L=R.count;for(let q=D,Y=D+L;q<Y;q+=3)f(e.getX(q+0),e.getX(q+1),e.getX(q+2))}let S=new z,_=new z,T=new z,b=new z;function w(x){T.fromBufferAttribute(s,x),b.copy(T);let E=o[x];S.copy(E),S.sub(T.multiplyScalar(T.dot(E))).normalize(),_.crossVectors(b,E);let D=_.dot(l[x])<0?-1:1;a.setXYZW(x,S.x,S.y,S.z,D)}for(let x=0,E=g.length;x<E;++x){let R=g[x],D=R.start,L=R.count;for(let q=D,Y=D+L;q<Y;q+=3)w(e.getX(q+0)),w(e.getX(q+1)),w(e.getX(q+2))}this._transformed=!0}computeVertexNormals(){let e=this.index,n=this.getAttribute("position");if(n!==void 0){let i=this.getAttribute("normal");if(i===void 0||i.count!==n.count)i=new An(new Float32Array(n.count*3),3),this.setAttribute("normal",i);else for(let u=0,d=i.count;u<d;u++)i.setXYZ(u,0,0,0);let s=new z,r=new z,a=new z,o=new z,l=new z,c=new z,h=new z,p=new z;if(e)for(let u=0,d=e.count;u<d;u+=3){let v=e.getX(u+0),M=e.getX(u+1),m=e.getX(u+2);s.fromBufferAttribute(n,v),r.fromBufferAttribute(n,M),a.fromBufferAttribute(n,m),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),o.fromBufferAttribute(i,v),l.fromBufferAttribute(i,M),c.fromBufferAttribute(i,m),o.add(h),l.add(h),c.add(h),i.setXYZ(v,o.x,o.y,o.z),i.setXYZ(M,l.x,l.y,l.z),i.setXYZ(m,c.x,c.y,c.z)}else for(let u=0,d=n.count;u<d;u+=3)s.fromBufferAttribute(n,u+0),r.fromBufferAttribute(n,u+1),a.fromBufferAttribute(n,u+2),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),i.setXYZ(u+0,h.x,h.y,h.z),i.setXYZ(u+1,h.x,h.y,h.z),i.setXYZ(u+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let n=0,i=e.count;n<i;n++)cn.fromBufferAttribute(e,n),cn.normalize(),e.setXYZ(n,cn.x,cn.y,cn.z)}toNonIndexed(){function e(o,l){let c=o.array,h=o.itemSize,p=o.normalized,u=new c.constructor(l.length*h),d=0,v=0;for(let M=0,m=l.length;M<m;M++){o.isInterleavedBufferAttribute?d=l[M]*o.data.stride+o.offset:d=l[M]*h;for(let f=0;f<h;f++)u[v++]=c[d++]}return new An(u,h,p)}if(this.index===null)return Re("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let n=new t,i=this.index.array,s=this.attributes;for(let o in s){let l=s[o],c=e(l,i);n.setAttribute(o,c)}let r=this.morphAttributes;for(let o in r){let l=[],c=r[o];for(let h=0,p=c.length;h<p;h++){let u=c[h],d=e(u,i);l.push(d)}n.morphAttributes[o]=l}n.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,l=a.length;o<l;o++){let c=a[o];n.addGroup(c.start,c.count,c.materialIndex)}return n}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let l=this.parameters;for(let c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};let n=this.index;n!==null&&(e.data.index={type:n.array.constructor.name,array:Array.prototype.slice.call(n.array)});let i=this.attributes;for(let l in i){let c=i[l];e.data.attributes[l]=c.toJSON(e.data)}let s={},r=!1;for(let l in this.morphAttributes){let c=this.morphAttributes[l],h=[];for(let p=0,u=c.length;p<u;p++){let d=c[p];h.push(d.toJSON(e.data))}h.length>0&&(s[l]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let n={};this.name=e.name;let i=e.index;i!==null&&this.setIndex(i.clone());let s=e.attributes;for(let c in s){let h=s[c];this.setAttribute(c,h.clone(n))}let r=e.morphAttributes;for(let c in r){let h=[],p=r[c];for(let u=0,d=p.length;u<d;u++)h.push(p[u].clone(n));this.morphAttributes[c]=h}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let c=0,h=a.length;c<h;c++){let p=a[c];this.addGroup(p.start,p.count,p.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}};var mT=0,Ci=class extends zn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:mT++}),this.uuid=sc(),this.name="",this.type="Material",this.blending=$r,this.side=wi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=cf,this.blendDst=uf,this.blendEquation=fr,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new ke(0,0,0),this.blendAlpha=0,this.depthFunc=ea,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=tg,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Kr,this.stencilZFail=Kr,this.stencilZPass=Kr,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let n in e){let i=e[n];if(i===void 0){Re(`Material: parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){Re(`Material: '${n}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector2&&i&&i.isVector2||s&&s.isEuler&&i&&i.isEuler||s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";n&&(e={textures:{},images:{}});let i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==$r&&(i.blending=this.blending),this.side!==wi&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==cf&&(i.blendSrc=this.blendSrc),this.blendDst!==uf&&(i.blendDst=this.blendDst),this.blendEquation!==fr&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==ea&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==tg&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Kr&&(i.stencilFail=this.stencilFail),this.stencilZFail!==Kr&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==Kr&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.allowOverride===!1&&(i.allowOverride=!1),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let a=[];for(let o in r){let l=r[o];delete l.metadata,a.push(l)}return a}if(n){let r=s(e.textures),a=s(e.images);r.length>0&&(i.textures=r),a.length>0&&(i.images=a)}return i}fromJSON(e,n){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new ke().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessRange=e.iridescenceThicknessRange),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=n[e.map]||null),e.matcap!==void 0&&(this.matcap=n[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=n[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=n[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=n[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let i=e.normalScale;Array.isArray(i)===!1&&(i=[i,i]),this.normalScale=new Ie().fromArray(i)}return e.displacementMap!==void 0&&(this.displacementMap=n[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=n[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=n[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=n[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=n[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=n[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=n[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=n[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=n[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=n[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=n[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=n[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=n[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=n[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new Ie().fromArray(e.clearcoatNormalScale)),e.iridescenceMap!==void 0&&(this.iridescenceMap=n[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=n[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=n[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=n[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=n[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=n[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=n[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let n=e.clippingPlanes,i=null;if(n!==null){let s=n.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=n[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}};var Ms=new z,Ym=new z,Qu=new z,lr=new z,qm=new z,Zu=new z,Qm=new z,Ef=class{constructor(e=new z,n=new z(0,0,-1)){this.origin=e,this.direction=n}set(e,n){return this.origin.copy(e),this.direction.copy(n),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,n){return n.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Ms)),this}closestPointToPoint(e,n){n.subVectors(e,this.origin);let i=n.dot(this.direction);return i<0?n.copy(this.origin):n.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let n=Ms.subVectors(e,this.origin).dot(this.direction);return n<0?this.origin.distanceToSquared(e):(Ms.copy(this.origin).addScaledVector(this.direction,n),Ms.distanceToSquared(e))}distanceSqToSegment(e,n,i,s){Ym.copy(e).add(n).multiplyScalar(.5),Qu.copy(n).sub(e).normalize(),lr.copy(this.origin).sub(Ym);let r=e.distanceTo(n)*.5,a=-this.direction.dot(Qu),o=lr.dot(this.direction),l=-lr.dot(Qu),c=lr.lengthSq(),h=Math.abs(1-a*a),p,u,d,v;if(h>0)if(p=a*l-o,u=a*o-l,v=r*h,p>=0)if(u>=-v)if(u<=v){let M=1/h;p*=M,u*=M,d=p*(p+a*u+2*o)+u*(a*p+u+2*l)+c}else u=r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u=-r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u<=-v?(p=Math.max(0,-(-a*r+o)),u=p>0?-r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c):u<=v?(p=0,u=Math.min(Math.max(-r,-l),r),d=u*(u+2*l)+c):(p=Math.max(0,-(a*r+o)),u=p>0?r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c);else u=a>0?-r:r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;return i&&i.copy(this.origin).addScaledVector(this.direction,p),s&&s.copy(Ym).addScaledVector(Qu,u),d}intersectSphere(e,n){Ms.subVectors(e.center,this.origin);let i=Ms.dot(this.direction),s=Ms.dot(Ms)-i*i,r=e.radius*e.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=i-a,l=i+a;return l<0?null:o<0?this.at(l,n):this.at(o,n)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let n=e.normal.dot(this.direction);if(n===0)return e.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(e.normal)+e.constant)/n;return i>=0?i:null}intersectPlane(e,n){let i=this.distanceToPlane(e);return i===null?null:this.at(i,n)}intersectsPlane(e){let n=e.distanceToPoint(this.origin);return n===0||e.normal.dot(this.direction)*n<0}intersectBox(e,n){let i,s,r,a,o,l,c=1/this.direction.x,h=1/this.direction.y,p=1/this.direction.z,u=this.origin;return c>=0?(i=(e.min.x-u.x)*c,s=(e.max.x-u.x)*c):(i=(e.max.x-u.x)*c,s=(e.min.x-u.x)*c),h>=0?(r=(e.min.y-u.y)*h,a=(e.max.y-u.y)*h):(r=(e.max.y-u.y)*h,a=(e.min.y-u.y)*h),i>a||r>s||((r>i||isNaN(i))&&(i=r),(a<s||isNaN(s))&&(s=a),p>=0?(o=(e.min.z-u.z)*p,l=(e.max.z-u.z)*p):(o=(e.max.z-u.z)*p,l=(e.min.z-u.z)*p),i>l||o>s)||((o>i||i!==i)&&(i=o),(l<s||s!==s)&&(s=l),s<0)?null:this.at(i>=0?i:s,n)}intersectsBox(e){return this.intersectBox(e,Ms)!==null}intersectTriangle(e,n,i,s,r){qm.subVectors(n,e),Zu.subVectors(i,e),Qm.crossVectors(qm,Zu);let a=this.direction.dot(Qm),o;if(a>0){if(s)return null;o=1}else if(a<0)o=-1,a=-a;else return null;lr.subVectors(this.origin,e);let l=o*this.direction.dot(Zu.crossVectors(lr,Zu));if(l<0)return null;let c=o*this.direction.dot(qm.cross(lr));if(c<0||l+c>a)return null;let h=-o*lr.dot(Qm);return h<0?null:this.at(h/a,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},Hl=class extends Ci{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new ke(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new hr,this.combine=cg,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},gS=new Ht,Qr=new Ef,Ku=new yo,vS=new z,Ju=new z,ju=new z,$u=new z,Zm=new z,ef=new z,xS=new z,tf=new z,Mn=class extends vi{constructor(e=new xi,n=new Hl){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){let s=n[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(e,n){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,a=i.morphTargetsRelative;n.fromBufferAttribute(s,e);let o=this.morphTargetInfluences;if(r&&o){ef.set(0,0,0);for(let l=0,c=r.length;l<c;l++){let h=o[l],p=r[l];h!==0&&(Zm.fromBufferAttribute(p,e),a?ef.addScaledVector(Zm,h):ef.addScaledVector(Zm.sub(n),h))}n.add(ef)}return n}raycast(e,n){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Ku.copy(i.boundingSphere),Ku.applyMatrix4(r),Qr.copy(e.ray).recast(e.near),!(Ku.containsPoint(Qr.origin)===!1&&(Qr.intersectSphere(Ku,vS)===null||Qr.origin.distanceToSquared(vS)>(e.far-e.near)**2))&&(gS.copy(r).invert(),Qr.copy(e.ray).applyMatrix4(gS),!(i.boundingBox!==null&&Qr.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,n,Qr)))}_computeIntersections(e,n,i){let s,r=this.geometry,a=this.material,o=r.index,l=r.attributes.position,c=r.attributes.uv,h=r.attributes.uv1,p=r.attributes.normal,u=r.groups,d=r.drawRange;if(o!==null)if(Array.isArray(a))for(let v=0,M=u.length;v<M;v++){let m=u[v],f=a[m.materialIndex],g=Math.max(m.start,d.start),S=Math.min(o.count,Math.min(m.start+m.count,d.start+d.count));for(let _=g,T=S;_<T;_+=3){let b=o.getX(_),w=o.getX(_+1),x=o.getX(_+2);s=nf(this,f,e,i,c,h,p,b,w,x),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=m.materialIndex,n.push(s))}}else{let v=Math.max(0,d.start),M=Math.min(o.count,d.start+d.count);for(let m=v,f=M;m<f;m+=3){let g=o.getX(m),S=o.getX(m+1),_=o.getX(m+2);s=nf(this,a,e,i,c,h,p,g,S,_),s&&(s.faceIndex=Math.floor(m/3),n.push(s))}}else if(l!==void 0)if(Array.isArray(a))for(let v=0,M=u.length;v<M;v++){let m=u[v],f=a[m.materialIndex],g=Math.max(m.start,d.start),S=Math.min(l.count,Math.min(m.start+m.count,d.start+d.count));for(let _=g,T=S;_<T;_+=3){let b=_,w=_+1,x=_+2;s=nf(this,f,e,i,c,h,p,b,w,x),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=m.materialIndex,n.push(s))}}else{let v=Math.max(0,d.start),M=Math.min(l.count,d.start+d.count);for(let m=v,f=M;m<f;m+=3){let g=m,S=m+1,_=m+2;s=nf(this,a,e,i,c,h,p,g,S,_),s&&(s.faceIndex=Math.floor(m/3),n.push(s))}}}};function gT(t,e,n,i,s,r,a,o){let l;if(e.side===Jt?l=i.intersectTriangle(a,r,s,!0,o):l=i.intersectTriangle(s,r,a,e.side===wi,o),l===null)return null;tf.copy(o),tf.applyMatrix4(t.matrixWorld);let c=n.ray.origin.distanceTo(tf);return c<n.near||c>n.far?null:{distance:c,point:tf.clone(),object:t}}function nf(t,e,n,i,s,r,a,o,l,c){t.getVertexPosition(o,Ju),t.getVertexPosition(l,ju),t.getVertexPosition(c,$u);let h=gT(t,e,n,i,Ju,ju,$u,xS);if(h){let p=new z;ur.getBarycoord(xS,Ju,ju,$u,p),s&&(h.uv=ur.getInterpolatedAttribute(s,o,l,c,p,new Ie)),r&&(h.uv1=ur.getInterpolatedAttribute(r,o,l,c,p,new Ie)),a&&(h.normal=ur.getInterpolatedAttribute(a,o,l,c,p,new z),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let u={a:o,b:l,c,normal:new z,materialIndex:0};ur.getNormal(Ju,ju,$u,u.normal),h.face=u,h.barycoord=p}return h}var Tf=class extends Kt{constructor(e=null,n=1,i=1,s,r,a,o,l,c=un,h=un,p,u){super(null,a,o,l,c,h,s,r,p,u),this.isDataTexture=!0,this.image={data:e,width:n,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var Km=new z,vT=new z,xT=new Pe,Wi=class{constructor(e=new z(1,0,0),n=0){this.isPlane=!0,this.normal=e,this.constant=n}set(e,n){return this.normal.copy(e),this.constant=n,this}setComponents(e,n,i,s){return this.normal.set(e,n,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,n){return this.normal.copy(e),this.constant=-n.dot(this.normal),this}setFromCoplanarPoints(e,n,i){let s=Km.subVectors(i,n).cross(vT.subVectors(e,n)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,n){return n.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,n,i=!0){let s=e.delta(Km),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?n.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/r;return i===!0&&(a<0||a>1)?null:n.copy(e.start).addScaledVector(s,a)}intersectsLine(e){let n=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return n<0&&i>0||i<0&&n>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,n){let i=n||xT.getNormalMatrix(e),s=this.coplanarPoint(Km).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}},Zr=new yo,yT=new Ie(.5,.5),sf=new z,Vl=class{constructor(e=new Wi,n=new Wi,i=new Wi,s=new Wi,r=new Wi,a=new Wi){this.planes=[e,n,i,s,r,a]}set(e,n,i,s,r,a){let o=this.planes;return o[0].copy(e),o[1].copy(n),o[2].copy(i),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(e){let n=this.planes;for(let i=0;i<6;i++)n[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,n=bi,i=!1){let s=this.planes,r=e.elements,a=r[0],o=r[1],l=r[2],c=r[3],h=r[4],p=r[5],u=r[6],d=r[7],v=r[8],M=r[9],m=r[10],f=r[11],g=r[12],S=r[13],_=r[14],T=r[15];if(s[0].setComponents(c-a,d-h,f-v,T-g).normalize(),s[1].setComponents(c+a,d+h,f+v,T+g).normalize(),s[2].setComponents(c+o,d+p,f+M,T+S).normalize(),s[3].setComponents(c-o,d-p,f-M,T-S).normalize(),i)s[4].setComponents(l,u,m,_).normalize(),s[5].setComponents(c-l,d-u,f-m,T-_).normalize();else if(s[4].setComponents(c-l,d-u,f-m,T-_).normalize(),n===bi)s[5].setComponents(c+l,d+u,f+m,T+_).normalize();else if(n===Ll)s[5].setComponents(l,u,m,_).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+n);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),Zr.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let n=e.geometry;n.boundingSphere===null&&n.computeBoundingSphere(),Zr.copy(n.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(Zr)}intersectsSprite(e){Zr.center.set(0,0,0);let n=yT.distanceTo(e.center);return Zr.radius=.7071067811865476+n,Zr.applyMatrix4(e.matrixWorld),this.intersectsSphere(Zr)}intersectsSphere(e){let n=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(n[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){let n=this.planes;for(let i=0;i<6;i++){let s=n[i];if(sf.x=s.normal.x>0?e.max.x:e.min.x,sf.y=s.normal.y>0?e.max.y:e.min.y,sf.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(sf)<0)return!1}return!0}containsPoint(e){let n=this.planes;for(let i=0;i<6;i++)if(n[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};var kl=class extends Kt{constructor(e=[],n=xr,i,s,r,a,o,l,c,h){super(e,n,i,s,r,a,o,l,c,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}};var Ri=class extends Kt{constructor(e,n,i=Ui,s,r,a,o=un,l=un,c,h=Yi,p=1){if(h!==Yi&&h!==Zi)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let u={width:e,height:n,depth:p};super(u,s,r,a,o,l,h,i,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new vo(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let n=super.toJSON(e);return this.compareFunction!==null&&(n.compareFunction=this.compareFunction),n}},bf=class extends Ri{constructor(e,n=Ui,i=xr,s,r,a=un,o=un,l,c=Yi){let h={width:e,height:e,depth:1},p=[h,h,h,h,h,h];super(e,e,n,i,s,r,a,o,l,c),this.image=p,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},Wl=class extends Kt{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},_o=class t extends xi{constructor(e=1,n=1,i=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:n,depth:i,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let l=[],c=[],h=[],p=[],u=0,d=0;v("z","y","x",-1,-1,i,n,e,a,r,0),v("z","y","x",1,-1,i,n,-e,a,r,1),v("x","z","y",1,1,e,i,n,s,a,2),v("x","z","y",1,-1,e,i,-n,s,a,3),v("x","y","z",1,-1,e,n,i,s,r,4),v("x","y","z",-1,-1,e,n,-i,s,r,5),this.setIndex(l),this.setAttribute("position",new gi(c,3)),this.setAttribute("normal",new gi(h,3)),this.setAttribute("uv",new gi(p,2));function v(M,m,f,g,S,_,T,b,w,x,E){let R=_/w,D=T/x,L=_/2,q=T/2,Y=b/2,N=w+1,k=x+1,V=0,j=0,ee=new z;for(let se=0;se<k;se++){let he=se*D-q;for(let ve=0;ve<N;ve++){let Ke=ve*R-L;ee[M]=Ke*g,ee[m]=he*S,ee[f]=Y,c.push(ee.x,ee.y,ee.z),ee[M]=0,ee[m]=0,ee[f]=b>0?1:-1,h.push(ee.x,ee.y,ee.z),p.push(ve/w),p.push(1-se/x),V+=1}}for(let se=0;se<x;se++)for(let he=0;he<w;he++){let ve=u+he+N*se,Ke=u+he+N*(se+1),yt=u+(he+1)+N*(se+1),Je=u+(he+1)+N*se;l.push(ve,Ke,Je),l.push(Ke,yt,Je),j+=6}o.addGroup(d,j,E),d+=j,u+=V}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var ta=class t extends xi{constructor(e=1,n=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:n,widthSegments:i,heightSegments:s};let r=e/2,a=n/2,o=Math.floor(i),l=Math.floor(s),c=o+1,h=l+1,p=e/o,u=n/l,d=[],v=[],M=[],m=[];for(let f=0;f<h;f++){let g=f*u-a;for(let S=0;S<c;S++){let _=S*p-r;v.push(_,-g,0),M.push(0,0,1),m.push(S/o),m.push(1-f/l)}}for(let f=0;f<l;f++)for(let g=0;g<o;g++){let S=g+c*f,_=g+c*(f+1),T=g+1+c*(f+1),b=g+1+c*f;d.push(S,_,b),d.push(_,T,b)}this.setIndex(d),this.setAttribute("position",new gi(v,3)),this.setAttribute("normal",new gi(M,3)),this.setAttribute("uv",new gi(m,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.widthSegments,e.heightSegments)}};function ia(t){let e={};for(let n in t){e[n]={};for(let i in t[n]){let s=t[n][i];if(yS(s))s.isRenderTargetTexture?(Re("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[n][i]=null):e[n][i]=s.clone();else if(Array.isArray(s))if(yS(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();e[n][i]=r}else e[n][i]=s.slice();else e[n][i]=s}}return e}function En(t){let e={};for(let n=0;n<t.length;n++){let i=ia(t[n]);for(let s in i)e[s]=i[s]}return e}function yS(t){return t&&(t.isColor||t.isMatrix3||t.isMatrix4||t.isVector2||t.isVector3||t.isVector4||t.isTexture||t.isQuaternion)}function _T(t){let e=[];for(let n=0;n<t.length;n++)e.push(t[n].clone());return e}function wg(t){let e=t.getRenderTarget();return e===null?t.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:Ye.workingColorSpace}var sA={clone:ia,merge:En},ST=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,AT=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,Vt=class extends Ci{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=ST,this.fragmentShader=AT,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=ia(e.uniforms),this.uniformsGroups=_T(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let n=super.toJSON(e);n.glslVersion=this.glslVersion,n.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?n.uniforms[s]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?n.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?n.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?n.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?n.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?n.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?n.uniforms[s]={type:"m4",value:a.toArray()}:n.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(n.defines=this.defines),n.vertexShader=this.vertexShader,n.fragmentShader=this.fragmentShader,n.lights=this.lights,n.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(n.extensions=i),n}fromJSON(e,n){if(super.fromJSON(e,n),e.uniforms!==void 0)for(let i in e.uniforms){let s=e.uniforms[i];switch(this.uniforms[i]={},s.type){case"t":this.uniforms[i].value=n[s.value]||null;break;case"c":this.uniforms[i].value=new ke().setHex(s.value);break;case"v2":this.uniforms[i].value=new Ie().fromArray(s.value);break;case"v3":this.uniforms[i].value=new z().fromArray(s.value);break;case"v4":this.uniforms[i].value=new Dt().fromArray(s.value);break;case"m3":this.uniforms[i].value=new Pe().fromArray(s.value);break;case"m4":this.uniforms[i].value=new Ht().fromArray(s.value);break;default:this.uniforms[i].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let i in e.extensions)this.extensions[i]=e.extensions[i];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},wf=class extends Vt{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}};var Cf=class extends Ci{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Ki,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},Rf=class extends Ci{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function rf(t,e){return!t||t.constructor===e?t:typeof e.BYTES_PER_ELEMENT=="number"?new e(t):Array.prototype.slice.call(t)}var mr=class{constructor(e,n,i,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new n.constructor(i),this.sampleValues=n,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(e){let n=this.parameterPositions,i=this._cachedIndex,s=n[i],r=n[i-1];e:{t:{let a;n:{i:if(!(e<s)){for(let o=i+2;;){if(s===void 0){if(e<r)break i;return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===o)break;if(r=s,s=n[++i],e<s)break t}a=n.length;break n}if(!(e>=r)){let o=n[1];e<o&&(i=2,r=o);for(let l=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===l)break;if(s=r,r=n[--i-1],e>=r)break t}a=i,i=0;break n}break e}for(;i<a;){let o=i+a>>>1;e<n[o]?a=o:i=o+1}if(s=n[i],r=n[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let n=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=e*s;for(let a=0;a!==s;++a)n[a]=i[r+a];return n}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},Df=class extends mr{constructor(e,n,i,s){super(e,n,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:jm,endingEnd:jm}}intervalChanged_(e,n,i){let s=this.parameterPositions,r=e-2,a=e+1,o=s[r],l=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case $m:r=e,o=2*n-i;break;case eg:r=s.length-2,o=n+s[r]-s[r+1];break;default:r=e,o=i}if(l===void 0)switch(this.getSettings_().endingEnd){case $m:a=e,l=2*i-n;break;case eg:a=1,l=i+s[1]-s[0];break;default:a=e-1,l=n}let c=(i-n)*.5,h=this.valueSize;this._weightPrev=c/(n-o),this._weightNext=c/(l-i),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this._offsetPrev,p=this._offsetNext,u=this._weightPrev,d=this._weightNext,v=(i-n)/(s-n),M=v*v,m=M*v,f=-u*m+2*u*M-u*v,g=(1+u)*m+(-1.5-2*u)*M+(-.5+u)*v+1,S=(-1-d)*m+(1.5+d)*M+.5*v,_=d*m-d*M;for(let T=0;T!==o;++T)r[T]=f*a[h+T]+g*a[c+T]+S*a[l+T]+_*a[p+T];return r}},Uf=class extends mr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=(i-n)/(s-n),p=1-h;for(let u=0;u!==o;++u)r[u]=a[c+u]*p+a[l+u]*h;return r}},Bf=class extends mr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e){return this.copySampleValue_(e-1)}},If=class extends mr{interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this.inTangents,p=this.outTangents;if(!h||!p){let v=(i-n)/(s-n),M=1-v;for(let m=0;m!==o;++m)r[m]=a[c+m]*M+a[l+m]*v;return r}let u=o*2,d=e-1;for(let v=0;v!==o;++v){let M=a[c+v],m=a[l+v],f=d*u+v*2,g=p[f],S=p[f+1],_=e*u+v*2,T=h[_],b=h[_+1],w=(i-n)/(s-n),x,E,R,D,L;for(let q=0;q<8;q++){x=w*w,E=x*w,R=1-w,D=R*R,L=D*R;let N=L*n+3*D*w*g+3*R*x*T+E*s-i;if(Math.abs(N)<1e-10)break;let k=3*D*(g-n)+6*R*w*(T-g)+3*x*(s-T);if(Math.abs(k)<1e-10)break;w=w-N/k,w=Math.max(0,Math.min(1,w))}r[v]=L*M+3*D*w*S+3*R*x*b+E*m}return r}},ei=class{constructor(e,n,i,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(n===void 0||n.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=rf(n,this.TimeBufferType),this.values=rf(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let n=e.constructor,i;if(n.toJSON!==this.toJSON)i=n.toJSON(e);else{i={name:e.name,times:rf(e.times,Array),values:rf(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(i.interpolation=s)}return i.type=e.ValueTypeName,i}InterpolantFactoryMethodDiscrete(e){return new Bf(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new Uf(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new Df(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let n=new If(this.times,this.values,this.getValueSize(),e);return this.settings&&(n.inTangents=this.settings.inTangents,n.outTangents=this.settings.outTangents),n}setInterpolation(e){let n;switch(e){case Il:n=this.InterpolantFactoryMethodDiscrete;break;case yf:n=this.InterpolantFactoryMethodLinear;break;case lf:n=this.InterpolantFactoryMethodSmooth;break;case Jm:n=this.InterpolantFactoryMethodBezier;break}if(n===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return Re("KeyframeTrack:",i),this}return this.createInterpolant=n,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return Il;case this.InterpolantFactoryMethodLinear:return yf;case this.InterpolantFactoryMethodSmooth:return lf;case this.InterpolantFactoryMethodBezier:return Jm}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]+=e}return this}scale(e){if(e!==1){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]*=e}return this}trim(e,n){let i=this.times,s=i.length,r=0,a=s-1;for(;r!==s&&i[r]<e;)++r;for(;a!==-1&&i[a]>n;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=i.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let e=!0,n=this.getValueSize();n-Math.floor(n)!==0&&(Ue("KeyframeTrack: Invalid value size in track.",this),e=!1);let i=this.times,s=this.values,r=i.length;r===0&&(Ue("KeyframeTrack: Track is empty.",this),e=!1);let a=null;for(let o=0;o!==r;o++){let l=i[o];if(typeof l=="number"&&isNaN(l)){Ue("KeyframeTrack: Time is not a valid number.",this,o,l),e=!1;break}if(a!==null&&a>l){Ue("KeyframeTrack: Out of order keys.",this,o,l,a),e=!1;break}a=l}if(s!==void 0&&eT(s))for(let o=0,l=s.length;o!==l;++o){let c=s[o];if(isNaN(c)){Ue("KeyframeTrack: Value is not a valid number.",this,o,c),e=!1;break}}return e}optimize(){let e=this.times.slice(),n=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===lf,r=e.length-1,a=1;for(let o=1;o<r;++o){let l=!1,c=e[o],h=e[o+1];if(c!==h&&(o!==1||c!==e[0]))if(s)l=!0;else{let p=o*i,u=p-i,d=p+i;for(let v=0;v!==i;++v){let M=n[p+v];if(M!==n[u+v]||M!==n[d+v]){l=!0;break}}}if(l){if(o!==a){e[a]=e[o];let p=o*i,u=a*i;for(let d=0;d!==i;++d)n[u+d]=n[p+d]}++a}}if(r>0){e[a]=e[r];for(let o=r*i,l=a*i,c=0;c!==i;++c)n[l+c]=n[o+c];++a}return a!==e.length?(this.times=e.slice(0,a),this.values=n.slice(0,a*i)):(this.times=e,this.values=n),this}clone(){let e=this.times.slice(),n=this.values.slice(),i=this.constructor,s=new i(this.name,e,n);return s.createInterpolant=this.createInterpolant,s}};ei.prototype.ValueTypeName="";ei.prototype.TimeBufferType=Float32Array;ei.prototype.ValueBufferType=Float32Array;ei.prototype.DefaultInterpolation=yf;var gr=class extends ei{constructor(e,n,i){super(e,n,i)}};gr.prototype.ValueTypeName="bool";gr.prototype.ValueBufferType=Array;gr.prototype.DefaultInterpolation=Il;gr.prototype.InterpolantFactoryMethodLinear=void 0;gr.prototype.InterpolantFactoryMethodSmooth=void 0;var Pf=class extends ei{constructor(e,n,i,s){super(e,n,i,s)}};Pf.prototype.ValueTypeName="color";var Lf=class extends ei{constructor(e,n,i,s){super(e,n,i,s)}};Lf.prototype.ValueTypeName="number";var Nf=class extends mr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=(i-n)/(s-n),c=e*o;for(let h=c+o;c!==h;c+=4)qi.slerpFlat(r,0,a,c-o,a,c,l);return r}},Xl=class extends ei{constructor(e,n,i,s){super(e,n,i,s)}InterpolantFactoryMethodLinear(e){return new Nf(this.times,this.values,this.getValueSize(),e)}};Xl.prototype.ValueTypeName="quaternion";Xl.prototype.InterpolantFactoryMethodSmooth=void 0;var vr=class extends ei{constructor(e,n,i){super(e,n,i)}};vr.prototype.ValueTypeName="string";vr.prototype.ValueBufferType=Array;vr.prototype.DefaultInterpolation=Il;vr.prototype.InterpolantFactoryMethodLinear=void 0;vr.prototype.InterpolantFactoryMethodSmooth=void 0;var Of=class extends ei{constructor(e,n,i,s){super(e,n,i,s)}};Of.prototype.ValueTypeName="vector";var Ff=class{constructor(e,n,i){let s=this,r=!1,a=0,o=0,l,c=[];this.onStart=void 0,this.onLoad=e,this.onProgress=n,this.onError=i,this._abortController=null,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),l?l(h):h},this.setURLModifier=function(h){return l=h,this},this.addHandler=function(h,p){return c.push(h,p),this},this.removeHandler=function(h){let p=c.indexOf(h);return p!==-1&&c.splice(p,2),this},this.getHandler=function(h){for(let p=0,u=c.length;p<u;p+=2){let d=c[p],v=c[p+1];if(d.global&&(d.lastIndex=0),d.test(h))return v}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},rA=new Ff,zf=class{constructor(e){this.manager=e!==void 0?e:rA,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,n){let i=this;return new Promise(function(s,r){i.load(e,s,n,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}};zf.DEFAULT_MATERIAL_NAME="__DEFAULT";var af=new z,of=new qi,ki=new z,Yl=class extends vi{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Ht,this.projectionMatrix=new Ht,this.projectionMatrixInverse=new Ht,this.coordinateSystem=bi,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,n){return super.copy(e,n),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(af,of,ki),ki.x===1&&ki.y===1&&ki.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(af,of,ki.set(1,1,1)).invert()}updateWorldMatrix(e,n,i=!1){super.updateWorldMatrix(e,n,i),this.matrixWorld.decompose(af,of,ki),ki.x===1&&ki.y===1&&ki.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(af,of,ki.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},cr=new z,_S=new Ie,SS=new Ie,Sn=class extends Yl{constructor(e=50,n=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=n,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let n=.5*this.getFilmHeight()/e;this.fov=_f*2*Math.atan(n),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(Cm*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return _f*2*Math.atan(Math.tan(Cm*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,n,i){cr.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(cr.x,cr.y).multiplyScalar(-e/cr.z),cr.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(cr.x,cr.y).multiplyScalar(-e/cr.z)}getViewSize(e,n){return this.getViewBounds(e,_S,SS),n.subVectors(SS,_S)}setViewOffset(e,n,i,s,r,a){this.aspect=e/n,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,n=e*Math.tan(Cm*.5*this.fov)/this.zoom,i=2*n,s=this.aspect*i,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let l=a.fullWidth,c=a.fullHeight;r+=a.offsetX*s/l,n-=a.offsetY*i/c,s*=a.width/l,i*=a.height/c}let o=this.filmOffset;o!==0&&(r+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,n,n-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.fov=this.fov,n.object.zoom=this.zoom,n.object.near=this.near,n.object.far=this.far,n.object.focus=this.focus,n.object.aspect=this.aspect,this.view!==null&&(n.object.view=Object.assign({},this.view)),n.object.filmGauge=this.filmGauge,n.object.filmOffset=this.filmOffset,n}};var bs=class extends Yl{constructor(e=-1,n=1,i=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=n,this.top=i,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,n,i,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),n=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-e,a=i+e,o=s+n,l=s-n;if(this.view!==null&&this.view.enabled){let c=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,a=r+c*this.view.width,o-=h*this.view.offsetY,l=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.zoom=this.zoom,n.object.left=this.left,n.object.right=this.right,n.object.top=this.top,n.object.bottom=this.bottom,n.object.near=this.near,n.object.far=this.far,this.view!==null&&(n.object.view=Object.assign({},this.view)),n}};var fo=-90,ho=1,Gf=class extends vi{constructor(e,n,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Sn(fo,ho,e,n);s.layers=this.layers,this.add(s);let r=new Sn(fo,ho,e,n);r.layers=this.layers,this.add(r);let a=new Sn(fo,ho,e,n);a.layers=this.layers,this.add(a);let o=new Sn(fo,ho,e,n);o.layers=this.layers,this.add(o);let l=new Sn(fo,ho,e,n);l.layers=this.layers,this.add(l);let c=new Sn(fo,ho,e,n);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){let e=this.coordinateSystem,n=this.children.concat(),[i,s,r,a,o,l]=n;for(let c of n)this.remove(c);if(e===bi)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===Ll)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let c of n)this.add(c),c.updateMatrixWorld()}update(e,n){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,l,c,h]=this.children,p=e.getRenderTarget(),u=e.getActiveCubeFace(),d=e.getActiveMipmapLevel(),v=e.xr.enabled;e.xr.enabled=!1;let M=i.texture.generateMipmaps;i.texture.generateMipmaps=!1;let m=!1;e.isWebGLRenderer===!0?m=e.state.buffers.depth.getReversed():m=e.reversedDepthBuffer,e.setRenderTarget(i,0,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,r),e.setRenderTarget(i,1,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,a),e.setRenderTarget(i,2,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,o),e.setRenderTarget(i,3,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,l),e.setRenderTarget(i,4,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,c),i.texture.generateMipmaps=M,e.setRenderTarget(i,5,s),m&&e.autoClear===!1&&e.clearDepth(),e.render(n,h),e.setRenderTarget(p,u,d),e.xr.enabled=v,i.texture.needsPMREMUpdate=!0}},Hf=class extends Sn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}};var Cg="\\[\\]\\.:\\/",MT=new RegExp("["+Cg+"]","g"),Rg="[^"+Cg+"]",ET="[^"+Cg.replace("\\.","")+"]",TT=/((?:WC+[\/:])*)/.source.replace("WC",Rg),bT=/(WCOD+)?/.source.replace("WCOD",ET),wT=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Rg),CT=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Rg),RT=new RegExp("^"+TT+bT+wT+CT+"$"),DT=["material","materials","bones","map"],ig=class{constructor(e,n,i){let s=i||Et.parseTrackName(n);this._targetGroup=e,this._bindings=e.subscribe_(n,s)}getValue(e,n){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(e,n)}setValue(e,n){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(e,n)}bind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].bind()}unbind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].unbind()}},Et=class t{constructor(e,n,i){this.path=n,this.parsedPath=i||t.parseTrackName(n),this.node=t.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,n,i){return e&&e.isAnimationObjectGroup?new t.Composite(e,n,i):new t(e,n,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(MT,"")}static parseTrackName(e){let n=RT.exec(e);if(n===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let i={nodeName:n[2],objectName:n[3],objectIndex:n[4],propertyName:n[5],propertyIndex:n[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);DT.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,n){if(n===void 0||n===""||n==="."||n===-1||n===e.name||n===e.uuid)return e;if(e.skeleton){let i=e.skeleton.getBoneByName(n);if(i!==void 0)return i}if(e.children){let i=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===n||o.uuid===n)return o;let l=i(o.children);if(l)return l}return null},s=i(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,n){e[n]=this.targetObject[this.propertyName]}_getValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)e[n++]=i[s]}_getValue_arrayElement(e,n){e[n]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,n){this.resolvedProperty.toArray(e,n)}_setValue_direct(e,n){this.targetObject[this.propertyName]=e[n]}_setValue_direct_setNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++]}_setValue_array_setNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,n){this.resolvedProperty[this.propertyIndex]=e[n]}_setValue_arrayElement_setNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,n){this.resolvedProperty.fromArray(e,n)}_setValue_fromArray_setNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,n){this.bind(),this.getValue(e,n)}_setValue_unbound(e,n){this.bind(),this.setValue(e,n)}bind(){let e=this.node,n=this.parsedPath,i=n.objectName,s=n.propertyName,r=n.propertyIndex;if(e||(e=t.findNode(this.rootNode,n.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){Re("PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let c=n.objectIndex;switch(i){case"materials":if(!e.material){Ue("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){Ue("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){Ue("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===c){c=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){Ue("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){Ue("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){Ue("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(c!==void 0){if(e[c]===void 0){Ue("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[c]}}let a=e[s];if(a===void 0){let c=n.nodeName;Ue("PropertyBinding: Trying to update property for track: "+c+"."+s+" but it wasn't found.",e);return}let o=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?o=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let l=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){Ue("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){Ue("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}l=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(l=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(l=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[l],this.setValue=this.SetterByBindingTypeAndVersioning[l][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Et.Composite=ig;Et.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Et.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Et.prototype.GetterByBindingType=[Et.prototype._getValue_direct,Et.prototype._getValue_array,Et.prototype._getValue_arrayElement,Et.prototype._getValue_toArray];Et.prototype.SetterByBindingTypeAndVersioning=[[Et.prototype._setValue_direct,Et.prototype._setValue_direct_setNeedsUpdate,Et.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_array,Et.prototype._setValue_array_setNeedsUpdate,Et.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_arrayElement,Et.prototype._setValue_arrayElement_setNeedsUpdate,Et.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_fromArray,Et.prototype._setValue_fromArray_setNeedsUpdate,Et.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var UD=new Float32Array(1);var Ct=class t{constructor(e){this.value=e}clone(){return new t(this.value.clone===void 0?this.value:this.value.clone())}};var ql=class{constructor(e=!0){this.autoStart=e,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1,Re("Clock: This module has been deprecated. Please use THREE.Timer instead.")}start(){this.startTime=performance.now(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let e=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){let n=performance.now();e=(n-this.oldTime)/1e3,this.oldTime=n,this.elapsedTime+=e}return e}};var Lg=class Lg{constructor(e,n,i,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,n,i,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,n=0){for(let i=0;i<4;i++)this.elements[i]=e[i+n];return this}set(e,n,i,s){let r=this.elements;return r[0]=e,r[2]=n,r[1]=i,r[3]=s,this}};Lg.prototype.isMatrix2=!0;var sg=Lg;function Dg(t,e,n,i){let s=UT(i);switch(n){case Ag:return t*e;case Eg:return t*e/s.components*s.byteLength;case Zf:return t*e/s.components*s.byteLength;case Sr:return t*e*2/s.components*s.byteLength;case Kf:return t*e*2/s.components*s.byteLength;case Mg:return t*e*3/s.components*s.byteLength;case yi:return t*e*4/s.components*s.byteLength;case Jf:return t*e*4/s.components*s.byteLength;case Jl:case jl:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case $l:case ec:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case $f:case th:return Math.max(t,16)*Math.max(e,8)/4;case jf:case eh:return Math.max(t,8)*Math.max(e,8)/2;case nh:case ih:case rh:case ah:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case sh:case tc:case oh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case lh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case ch:return Math.floor((t+4)/5)*Math.floor((e+3)/4)*16;case uh:return Math.floor((t+4)/5)*Math.floor((e+4)/5)*16;case fh:return Math.floor((t+5)/6)*Math.floor((e+4)/5)*16;case hh:return Math.floor((t+5)/6)*Math.floor((e+5)/6)*16;case dh:return Math.floor((t+7)/8)*Math.floor((e+4)/5)*16;case ph:return Math.floor((t+7)/8)*Math.floor((e+5)/6)*16;case mh:return Math.floor((t+7)/8)*Math.floor((e+7)/8)*16;case gh:return Math.floor((t+9)/10)*Math.floor((e+4)/5)*16;case vh:return Math.floor((t+9)/10)*Math.floor((e+5)/6)*16;case xh:return Math.floor((t+9)/10)*Math.floor((e+7)/8)*16;case yh:return Math.floor((t+9)/10)*Math.floor((e+9)/10)*16;case _h:return Math.floor((t+11)/12)*Math.floor((e+9)/10)*16;case Sh:return Math.floor((t+11)/12)*Math.floor((e+11)/12)*16;case Ah:case Mh:case Eh:return Math.ceil(t/4)*Math.ceil(e/4)*16;case Th:case bh:return Math.ceil(t/4)*Math.ceil(e/4)*8;case nc:case wh:return Math.ceil(t/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${n} format.`)}function UT(t){switch(t){case jt:case xg:return{byteLength:1,components:1};case Ao:case yg:case Qi:return{byteLength:2,components:1};case qf:case Qf:return{byteLength:2,components:4};case Ui:case Yf:case ti:return{byteLength:4,components:1};case _g:case Sg:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${t}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"185"}}));typeof window<"u"&&(window.__THREE__?Re("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="185");function CA(){let t=null,e=!1,n=null,i=null;function s(r,a){n(r,a),i=t.requestAnimationFrame(s)}return{start:function(){e!==!0&&n!==null&&t!==null&&(i=t.requestAnimationFrame(s),e=!0)},stop:function(){t!==null&&t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){n=r},setContext:function(r){t=r}}}function BT(t){let e=new WeakMap;function n(o,l){let c=o.array,h=o.usage,p=c.byteLength,u=t.createBuffer();t.bindBuffer(l,u),t.bufferData(l,c,h),o.onUploadCallback();let d;if(c instanceof Float32Array)d=t.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)d=t.HALF_FLOAT;else if(c instanceof Uint16Array)o.isFloat16BufferAttribute?d=t.HALF_FLOAT:d=t.UNSIGNED_SHORT;else if(c instanceof Int16Array)d=t.SHORT;else if(c instanceof Uint32Array)d=t.UNSIGNED_INT;else if(c instanceof Int32Array)d=t.INT;else if(c instanceof Int8Array)d=t.BYTE;else if(c instanceof Uint8Array)d=t.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)d=t.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:u,type:d,bytesPerElement:c.BYTES_PER_ELEMENT,version:o.version,size:p}}function i(o,l,c){let h=l.array,p=l.updateRanges;if(t.bindBuffer(c,o),p.length===0)t.bufferSubData(c,0,h);else{p.sort((d,v)=>d.start-v.start);let u=0;for(let d=1;d<p.length;d++){let v=p[u],M=p[d];M.start<=v.start+v.count+1?v.count=Math.max(v.count,M.start+M.count-v.start):(++u,p[u]=M)}p.length=u+1;for(let d=0,v=p.length;d<v;d++){let M=p[d];t.bufferSubData(c,M.start*h.BYTES_PER_ELEMENT,h,M.start,M.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let l=e.get(o);l&&(t.deleteBuffer(l.buffer),e.delete(o))}function a(o,l){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let c=e.get(o);if(c===void 0)e.set(o,n(o,l));else if(c.version<o.version){if(c.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(c.buffer,o,l),c.version=o.version}}return{get:s,remove:r,update:a}}var IT=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,PT=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,LT=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,NT=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,OT=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,FT=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,zT=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,GT=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,HT=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,VT=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,kT=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,WT=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,XT=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,YT=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,qT=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,QT=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,ZT=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,KT=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,JT=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,jT=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,$T=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,eb=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,tb=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,nb=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
#define inverseTransformDirection transformDirectionByInverseViewMatrix
vec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {
	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );
}
vec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {
	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,ib=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,sb=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
#endif`,rb=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,ab=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,ob=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,lb=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,cb="gl_FragColor = linearToOutputTexel( gl_FragColor );",ub=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,fb=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,hb=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,db=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,pb=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,mb=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,gb=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,vb=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,xb=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,yb=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,_b=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,Sb=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Ab=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,Mb=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Eb=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,Tb=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,bb=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,wb=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Cb=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Rb=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Db=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Ub=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Bb=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#ifdef USE_LIGHT_PROBES_GRID
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		irradiance += getLightProbeGridIrradiance( probeWorldPos, probeWorldNormal );
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,Ib=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Pb=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Lb=`#ifdef USE_LIGHT_PROBES_GRID
uniform highp sampler3D probesSH;
uniform vec3 probesMin;
uniform vec3 probesMax;
uniform vec3 probesResolution;
vec3 getLightProbeGridIrradiance( vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 resMinusOne = res - 1.0;
	vec3 probeSpacing = gridRange / resMinusOne;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.0, 1.0 );
	uvw = uvw * resMinusOne / res + 0.5 / res;
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,Nb=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Ob=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Fb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,zb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Gb=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Hb=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Vb=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,kb=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Wb=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Xb=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Yb=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,qb=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Qb=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Zb=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,Kb=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Jb=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#ifdef DOUBLE_SIDED
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#ifdef DOUBLE_SIDED
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,jb=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,$b=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,ew=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,tw=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,nw=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,iw=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,sw=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,rw=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,aw=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,ow=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,lw=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,cw=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,uw=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,fw=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,hw=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,dw=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,pw=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,mw=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,gw=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,vw=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,xw=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,yw=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,_w=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,Sw=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,Aw=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,Mw=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Ew=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,Tw=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,bw=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,ww=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,Cw=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Rw=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Dw=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Uw=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Bw=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,Iw=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Pw=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Lw=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Nw=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ow=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Fw=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,zw=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Gw=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Hw=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Vw=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,kw=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Ww=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Xw=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Yw=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,qw=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Qw=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Zw=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Kw=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Jw=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,jw=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,$w=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,eC=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,tC=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,nC=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,iC=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,sC=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,rC=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,aC=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,oC=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,lC=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,cC=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,uC=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,fC=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,hC=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ze={alphahash_fragment:IT,alphahash_pars_fragment:PT,alphamap_fragment:LT,alphamap_pars_fragment:NT,alphatest_fragment:OT,alphatest_pars_fragment:FT,aomap_fragment:zT,aomap_pars_fragment:GT,batching_pars_vertex:HT,batching_vertex:VT,begin_vertex:kT,beginnormal_vertex:WT,bsdfs:XT,iridescence_fragment:YT,bumpmap_pars_fragment:qT,clipping_planes_fragment:QT,clipping_planes_pars_fragment:ZT,clipping_planes_pars_vertex:KT,clipping_planes_vertex:JT,color_fragment:jT,color_pars_fragment:$T,color_pars_vertex:eb,color_vertex:tb,common:nb,cube_uv_reflection_fragment:ib,defaultnormal_vertex:sb,displacementmap_pars_vertex:rb,displacementmap_vertex:ab,emissivemap_fragment:ob,emissivemap_pars_fragment:lb,colorspace_fragment:cb,colorspace_pars_fragment:ub,envmap_fragment:fb,envmap_common_pars_fragment:hb,envmap_pars_fragment:db,envmap_pars_vertex:pb,envmap_physical_pars_fragment:Tb,envmap_vertex:mb,fog_vertex:gb,fog_pars_vertex:vb,fog_fragment:xb,fog_pars_fragment:yb,gradientmap_pars_fragment:_b,lightmap_pars_fragment:Sb,lights_lambert_fragment:Ab,lights_lambert_pars_fragment:Mb,lights_pars_begin:Eb,lights_toon_fragment:bb,lights_toon_pars_fragment:wb,lights_phong_fragment:Cb,lights_phong_pars_fragment:Rb,lights_physical_fragment:Db,lights_physical_pars_fragment:Ub,lights_fragment_begin:Bb,lights_fragment_maps:Ib,lights_fragment_end:Pb,lightprobes_pars_fragment:Lb,logdepthbuf_fragment:Nb,logdepthbuf_pars_fragment:Ob,logdepthbuf_pars_vertex:Fb,logdepthbuf_vertex:zb,map_fragment:Gb,map_pars_fragment:Hb,map_particle_fragment:Vb,map_particle_pars_fragment:kb,metalnessmap_fragment:Wb,metalnessmap_pars_fragment:Xb,morphinstance_vertex:Yb,morphcolor_vertex:qb,morphnormal_vertex:Qb,morphtarget_pars_vertex:Zb,morphtarget_vertex:Kb,normal_fragment_begin:Jb,normal_fragment_maps:jb,normal_pars_fragment:$b,normal_pars_vertex:ew,normal_vertex:tw,normalmap_pars_fragment:nw,clearcoat_normal_fragment_begin:iw,clearcoat_normal_fragment_maps:sw,clearcoat_pars_fragment:rw,iridescence_pars_fragment:aw,opaque_fragment:ow,packing:lw,premultiplied_alpha_fragment:cw,project_vertex:uw,dithering_fragment:fw,dithering_pars_fragment:hw,roughnessmap_fragment:dw,roughnessmap_pars_fragment:pw,shadowmap_pars_fragment:mw,shadowmap_pars_vertex:gw,shadowmap_vertex:vw,shadowmask_pars_fragment:xw,skinbase_vertex:yw,skinning_pars_vertex:_w,skinning_vertex:Sw,skinnormal_vertex:Aw,specularmap_fragment:Mw,specularmap_pars_fragment:Ew,tonemapping_fragment:Tw,tonemapping_pars_fragment:bw,transmission_fragment:ww,transmission_pars_fragment:Cw,uv_pars_fragment:Rw,uv_pars_vertex:Dw,uv_vertex:Uw,worldpos_vertex:Bw,background_vert:Iw,background_frag:Pw,backgroundCube_vert:Lw,backgroundCube_frag:Nw,cube_vert:Ow,cube_frag:Fw,depth_vert:zw,depth_frag:Gw,distance_vert:Hw,distance_frag:Vw,equirect_vert:kw,equirect_frag:Ww,linedashed_vert:Xw,linedashed_frag:Yw,meshbasic_vert:qw,meshbasic_frag:Qw,meshlambert_vert:Zw,meshlambert_frag:Kw,meshmatcap_vert:Jw,meshmatcap_frag:jw,meshnormal_vert:$w,meshnormal_frag:eC,meshphong_vert:tC,meshphong_frag:nC,meshphysical_vert:iC,meshphysical_frag:sC,meshtoon_vert:rC,meshtoon_frag:aC,points_vert:oC,points_frag:lC,shadow_vert:cC,shadow_frag:uC,sprite_vert:fC,sprite_frag:hC},fe={common:{diffuse:{value:new ke(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Pe},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Pe}},envmap:{envMap:{value:null},envMapRotation:{value:new Pe},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Pe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Pe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Pe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Pe},normalScale:{value:new Ie(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Pe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Pe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Pe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Pe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new ke(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:null},probesMin:{value:new z},probesMax:{value:new z},probesResolution:{value:new z}},points:{diffuse:{value:new ke(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0},uvTransform:{value:new Pe}},sprite:{diffuse:{value:new ke(16777215)},opacity:{value:1},center:{value:new Ie(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Pe},alphaMap:{value:null},alphaMapTransform:{value:new Pe},alphaTest:{value:0}}},ji={basic:{uniforms:En([fe.common,fe.specularmap,fe.envmap,fe.aomap,fe.lightmap,fe.fog]),vertexShader:ze.meshbasic_vert,fragmentShader:ze.meshbasic_frag},lambert:{uniforms:En([fe.common,fe.specularmap,fe.envmap,fe.aomap,fe.lightmap,fe.emissivemap,fe.bumpmap,fe.normalmap,fe.displacementmap,fe.fog,fe.lights,{emissive:{value:new ke(0)},envMapIntensity:{value:1}}]),vertexShader:ze.meshlambert_vert,fragmentShader:ze.meshlambert_frag},phong:{uniforms:En([fe.common,fe.specularmap,fe.envmap,fe.aomap,fe.lightmap,fe.emissivemap,fe.bumpmap,fe.normalmap,fe.displacementmap,fe.fog,fe.lights,{emissive:{value:new ke(0)},specular:{value:new ke(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:ze.meshphong_vert,fragmentShader:ze.meshphong_frag},standard:{uniforms:En([fe.common,fe.envmap,fe.aomap,fe.lightmap,fe.emissivemap,fe.bumpmap,fe.normalmap,fe.displacementmap,fe.roughnessmap,fe.metalnessmap,fe.fog,fe.lights,{emissive:{value:new ke(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ze.meshphysical_vert,fragmentShader:ze.meshphysical_frag},toon:{uniforms:En([fe.common,fe.aomap,fe.lightmap,fe.emissivemap,fe.bumpmap,fe.normalmap,fe.displacementmap,fe.gradientmap,fe.fog,fe.lights,{emissive:{value:new ke(0)}}]),vertexShader:ze.meshtoon_vert,fragmentShader:ze.meshtoon_frag},matcap:{uniforms:En([fe.common,fe.bumpmap,fe.normalmap,fe.displacementmap,fe.fog,{matcap:{value:null}}]),vertexShader:ze.meshmatcap_vert,fragmentShader:ze.meshmatcap_frag},points:{uniforms:En([fe.points,fe.fog]),vertexShader:ze.points_vert,fragmentShader:ze.points_frag},dashed:{uniforms:En([fe.common,fe.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ze.linedashed_vert,fragmentShader:ze.linedashed_frag},depth:{uniforms:En([fe.common,fe.displacementmap]),vertexShader:ze.depth_vert,fragmentShader:ze.depth_frag},normal:{uniforms:En([fe.common,fe.bumpmap,fe.normalmap,fe.displacementmap,{opacity:{value:1}}]),vertexShader:ze.meshnormal_vert,fragmentShader:ze.meshnormal_frag},sprite:{uniforms:En([fe.sprite,fe.fog]),vertexShader:ze.sprite_vert,fragmentShader:ze.sprite_frag},background:{uniforms:{uvTransform:{value:new Pe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ze.background_vert,fragmentShader:ze.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Pe}},vertexShader:ze.backgroundCube_vert,fragmentShader:ze.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ze.cube_vert,fragmentShader:ze.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ze.equirect_vert,fragmentShader:ze.equirect_frag},distance:{uniforms:En([fe.common,fe.displacementmap,{referencePosition:{value:new z},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ze.distance_vert,fragmentShader:ze.distance_frag},shadow:{uniforms:En([fe.lights,fe.fog,{color:{value:new ke(0)},opacity:{value:1}}]),vertexShader:ze.shadow_vert,fragmentShader:ze.shadow_frag}};ji.physical={uniforms:En([ji.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Pe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Pe},clearcoatNormalScale:{value:new Ie(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Pe},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Pe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Pe},sheen:{value:0},sheenColor:{value:new ke(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Pe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Pe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Pe},transmissionSamplerSize:{value:new Ie},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Pe},attenuationDistance:{value:0},attenuationColor:{value:new ke(0)},specularColor:{value:new ke(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Pe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Pe},anisotropyVector:{value:new Ie},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Pe}}]),vertexShader:ze.meshphysical_vert,fragmentShader:ze.meshphysical_frag};var Dh={r:0,b:0,g:0},dC=new Ht,RA=new Pe;RA.set(-1,0,0,0,1,0,0,0,1);function pC(t,e,n,i,s,r){let a=new ke(0),o=s===!0?0:1,l,c,h=null,p=0,u=null;function d(g){let S=g.isScene===!0?g.background:null;if(S&&S.isTexture){let _=g.backgroundBlurriness>0;S=e.get(S,_)}return S}function v(g){let S=!1,_=d(g);_===null?m(a,o):_&&_.isColor&&(m(_,1),S=!0);let T=t.xr.getEnvironmentBlendMode();T==="additive"?n.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,r),(t.autoClear||S)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil))}function M(g,S){let _=d(S);_&&(_.isCubeTexture||_.mapping===Zl)?(c===void 0&&(c=new Mn(new _o(1,1,1),new Vt({name:"BackgroundCubeMaterial",uniforms:ia(ji.backgroundCube.uniforms),vertexShader:ji.backgroundCube.vertexShader,fragmentShader:ji.backgroundCube.fragmentShader,side:Jt,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),c.geometry.deleteAttribute("uv"),c.onBeforeRender=function(T,b,w){this.matrixWorld.copyPosition(w.matrixWorld)},Object.defineProperty(c.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(c)),c.material.uniforms.envMap.value=_,c.material.uniforms.backgroundBlurriness.value=S.backgroundBlurriness,c.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,c.material.uniforms.backgroundRotation.value.setFromMatrix4(dC.makeRotationFromEuler(S.backgroundRotation)).transpose(),_.isCubeTexture&&_.isRenderTargetTexture===!1&&c.material.uniforms.backgroundRotation.value.premultiply(RA),c.material.toneMapped=Ye.getTransfer(_.colorSpace)!==st,(h!==_||p!==_.version||u!==t.toneMapping)&&(c.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),c.layers.enableAll(),g.unshift(c,c.geometry,c.material,0,0,null)):_&&_.isTexture&&(l===void 0&&(l=new Mn(new ta(2,2),new Vt({name:"BackgroundMaterial",uniforms:ia(ji.background.uniforms),vertexShader:ji.background.vertexShader,fragmentShader:ji.background.fragmentShader,side:wi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(l)),l.material.uniforms.t2D.value=_,l.material.uniforms.backgroundIntensity.value=S.backgroundIntensity,l.material.toneMapped=Ye.getTransfer(_.colorSpace)!==st,_.matrixAutoUpdate===!0&&_.updateMatrix(),l.material.uniforms.uvTransform.value.copy(_.matrix),(h!==_||p!==_.version||u!==t.toneMapping)&&(l.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),l.layers.enableAll(),g.unshift(l,l.geometry,l.material,0,0,null))}function m(g,S){g.getRGB(Dh,wg(t)),n.buffers.color.setClear(Dh.r,Dh.g,Dh.b,S,r)}function f(){c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0),l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0)}return{getClearColor:function(){return a},setClearColor:function(g,S=1){a.set(g),o=S,m(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(g){o=g,m(a,o)},render:v,addToRenderList:M,dispose:f}}function mC(t,e){let n=t.getParameter(t.MAX_VERTEX_ATTRIBS),i={},s=u(null),r=s,a=!1;function o(D,L,q,Y,N){let k=!1,V=p(D,Y,q,L);r!==V&&(r=V,c(r.object)),k=d(D,Y,q,N),k&&v(D,Y,q,N),N!==null&&e.update(N,t.ELEMENT_ARRAY_BUFFER),(k||a)&&(a=!1,_(D,L,q,Y),N!==null&&t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,e.get(N).buffer))}function l(){return t.createVertexArray()}function c(D){return t.bindVertexArray(D)}function h(D){return t.deleteVertexArray(D)}function p(D,L,q,Y){let N=Y.wireframe===!0,k=i[L.id];k===void 0&&(k={},i[L.id]=k);let V=D.isInstancedMesh===!0?D.id:0,j=k[V];j===void 0&&(j={},k[V]=j);let ee=j[q.id];ee===void 0&&(ee={},j[q.id]=ee);let se=ee[N];return se===void 0&&(se=u(l()),ee[N]=se),se}function u(D){let L=[],q=[],Y=[];for(let N=0;N<n;N++)L[N]=0,q[N]=0,Y[N]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:L,enabledAttributes:q,attributeDivisors:Y,object:D,attributes:{},index:null}}function d(D,L,q,Y){let N=r.attributes,k=L.attributes,V=0,j=q.getAttributes();for(let ee in j)if(j[ee].location>=0){let he=N[ee],ve=k[ee];if(ve===void 0&&(ee==="instanceMatrix"&&D.instanceMatrix&&(ve=D.instanceMatrix),ee==="instanceColor"&&D.instanceColor&&(ve=D.instanceColor)),he===void 0||he.attribute!==ve||ve&&he.data!==ve.data)return!0;V++}return r.attributesNum!==V||r.index!==Y}function v(D,L,q,Y){let N={},k=L.attributes,V=0,j=q.getAttributes();for(let ee in j)if(j[ee].location>=0){let he=k[ee];he===void 0&&(ee==="instanceMatrix"&&D.instanceMatrix&&(he=D.instanceMatrix),ee==="instanceColor"&&D.instanceColor&&(he=D.instanceColor));let ve={};ve.attribute=he,he&&he.data&&(ve.data=he.data),N[ee]=ve,V++}r.attributes=N,r.attributesNum=V,r.index=Y}function M(){let D=r.newAttributes;for(let L=0,q=D.length;L<q;L++)D[L]=0}function m(D){f(D,0)}function f(D,L){let q=r.newAttributes,Y=r.enabledAttributes,N=r.attributeDivisors;q[D]=1,Y[D]===0&&(t.enableVertexAttribArray(D),Y[D]=1),N[D]!==L&&(t.vertexAttribDivisor(D,L),N[D]=L)}function g(){let D=r.newAttributes,L=r.enabledAttributes;for(let q=0,Y=L.length;q<Y;q++)L[q]!==D[q]&&(t.disableVertexAttribArray(q),L[q]=0)}function S(D,L,q,Y,N,k,V){V===!0?t.vertexAttribIPointer(D,L,q,N,k):t.vertexAttribPointer(D,L,q,Y,N,k)}function _(D,L,q,Y){M();let N=Y.attributes,k=q.getAttributes(),V=L.defaultAttributeValues;for(let j in k){let ee=k[j];if(ee.location>=0){let se=N[j];if(se===void 0&&(j==="instanceMatrix"&&D.instanceMatrix&&(se=D.instanceMatrix),j==="instanceColor"&&D.instanceColor&&(se=D.instanceColor)),se!==void 0){let he=se.normalized,ve=se.itemSize,Ke=e.get(se);if(Ke===void 0)continue;let yt=Ke.buffer,Je=Ke.type,Z=Ke.bytesPerElement,ie=Je===t.INT||Je===t.UNSIGNED_INT||se.gpuType===Yf;if(se.isInterleavedBufferAttribute){let te=se.data,Se=te.stride,Ae=se.offset;if(te.isInstancedInterleavedBuffer){for(let Ce=0;Ce<ee.locationSize;Ce++)f(ee.location+Ce,te.meshPerAttribute);D.isInstancedMesh!==!0&&Y._maxInstanceCount===void 0&&(Y._maxInstanceCount=te.meshPerAttribute*te.count)}else for(let Ce=0;Ce<ee.locationSize;Ce++)m(ee.location+Ce);t.bindBuffer(t.ARRAY_BUFFER,yt);for(let Ce=0;Ce<ee.locationSize;Ce++)S(ee.location+Ce,ve/ee.locationSize,Je,he,Se*Z,(Ae+ve/ee.locationSize*Ce)*Z,ie)}else{if(se.isInstancedBufferAttribute){for(let te=0;te<ee.locationSize;te++)f(ee.location+te,se.meshPerAttribute);D.isInstancedMesh!==!0&&Y._maxInstanceCount===void 0&&(Y._maxInstanceCount=se.meshPerAttribute*se.count)}else for(let te=0;te<ee.locationSize;te++)m(ee.location+te);t.bindBuffer(t.ARRAY_BUFFER,yt);for(let te=0;te<ee.locationSize;te++)S(ee.location+te,ve/ee.locationSize,Je,he,ve*Z,ve/ee.locationSize*te*Z,ie)}}else if(V!==void 0){let he=V[j];if(he!==void 0)switch(he.length){case 2:t.vertexAttrib2fv(ee.location,he);break;case 3:t.vertexAttrib3fv(ee.location,he);break;case 4:t.vertexAttrib4fv(ee.location,he);break;default:t.vertexAttrib1fv(ee.location,he)}}}}g()}function T(){E();for(let D in i){let L=i[D];for(let q in L){let Y=L[q];for(let N in Y){let k=Y[N];for(let V in k)h(k[V].object),delete k[V];delete Y[N]}}delete i[D]}}function b(D){if(i[D.id]===void 0)return;let L=i[D.id];for(let q in L){let Y=L[q];for(let N in Y){let k=Y[N];for(let V in k)h(k[V].object),delete k[V];delete Y[N]}}delete i[D.id]}function w(D){for(let L in i){let q=i[L];for(let Y in q){let N=q[Y];if(N[D.id]===void 0)continue;let k=N[D.id];for(let V in k)h(k[V].object),delete k[V];delete N[D.id]}}}function x(D){for(let L in i){let q=i[L],Y=D.isInstancedMesh===!0?D.id:0,N=q[Y];if(N!==void 0){for(let k in N){let V=N[k];for(let j in V)h(V[j].object),delete V[j];delete N[k]}delete q[Y],Object.keys(q).length===0&&delete i[L]}}}function E(){R(),a=!0,r!==s&&(r=s,c(r.object))}function R(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:E,resetDefaultState:R,dispose:T,releaseStatesOfGeometry:b,releaseStatesOfObject:x,releaseStatesOfProgram:w,initAttributes:M,enableAttribute:m,disableUnusedAttributes:g}}function gC(t,e,n){let i;function s(l){i=l}function r(l,c){t.drawArrays(i,l,c),n.update(c,i,1)}function a(l,c,h){h!==0&&(t.drawArraysInstanced(i,l,c,h),n.update(c,i,h))}function o(l,c,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,l,0,c,0,h);let u=0;for(let d=0;d<h;d++)u+=c[d];n.update(u,i,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function vC(t,e,n,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let w=e.get("EXT_texture_filter_anisotropic");s=t.getParameter(w.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(w){return!(w!==yi&&i.convert(w)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(w){let x=w===Qi&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(w!==jt&&i.convert(w)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_TYPE)&&w!==ti&&!x)}function l(w){if(w==="highp"){if(t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.HIGH_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.HIGH_FLOAT).precision>0)return"highp";w="mediump"}return w==="mediump"&&t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.MEDIUM_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=n.precision!==void 0?n.precision:"highp",h=l(c);h!==c&&(Re("WebGLRenderer:",c,"not supported, using",h,"instead."),c=h);let p=n.logarithmicDepthBuffer===!0,u=n.reversedDepthBuffer===!0&&e.has("EXT_clip_control");n.reversedDepthBuffer===!0&&u===!1&&Re("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer.");let d=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS),v=t.getParameter(t.MAX_VERTEX_TEXTURE_IMAGE_UNITS),M=t.getParameter(t.MAX_TEXTURE_SIZE),m=t.getParameter(t.MAX_CUBE_MAP_TEXTURE_SIZE),f=t.getParameter(t.MAX_VERTEX_ATTRIBS),g=t.getParameter(t.MAX_VERTEX_UNIFORM_VECTORS),S=t.getParameter(t.MAX_VARYING_VECTORS),_=t.getParameter(t.MAX_FRAGMENT_UNIFORM_VECTORS),T=t.getParameter(t.MAX_SAMPLES),b=t.getParameter(t.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:a,textureTypeReadable:o,precision:c,logarithmicDepthBuffer:p,reversedDepthBuffer:u,maxTextures:d,maxVertexTextures:v,maxTextureSize:M,maxCubemapSize:m,maxAttributes:f,maxVertexUniforms:g,maxVaryings:S,maxFragmentUniforms:_,maxSamples:T,samples:b}}function xC(t){let e=this,n=null,i=0,s=!1,r=!1,a=new Wi,o=new Pe,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(p,u){let d=p.length!==0||u||i!==0||s;return s=u,i=p.length,d},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(p,u){n=h(p,u,0)},this.setState=function(p,u,d){let v=p.clippingPlanes,M=p.clipIntersection,m=p.clipShadows,f=t.get(p);if(!s||v===null||v.length===0||r&&!m)r?h(null):c();else{let g=r?0:i,S=g*4,_=f.clippingState||null;l.value=_,_=h(v,u,S,d);for(let T=0;T!==S;++T)_[T]=n[T];f.clippingState=_,this.numIntersection=M?this.numPlanes:0,this.numPlanes+=g}};function c(){l.value!==n&&(l.value=n,l.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(p,u,d,v){let M=p!==null?p.length:0,m=null;if(M!==0){if(m=l.value,v!==!0||m===null){let f=d+M*4,g=u.matrixWorldInverse;o.getNormalMatrix(g),(m===null||m.length<f)&&(m=new Float32Array(f));for(let S=0,_=d;S!==M;++S,_+=4)a.copy(p[S]).applyMatrix4(g,o),a.normal.toArray(m,_),m[_+3]=a.constant}l.value=m,l.needsUpdate=!0}return e.numPlanes=M,e.numIntersection=0,m}}var Ar=4,aA=[.125,.215,.35,.446,.526,.582],sa=20,yC=256,rc=new bs,oA=new ke,Ng=null,Og=0,Fg=0,zg=!1,_C=new z,Bh=class{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,n=0,i=.1,s=100,r={}){let{size:a=256,position:o=_C}=r;Ng=this._renderer.getRenderTarget(),Og=this._renderer.getActiveCubeFace(),Fg=this._renderer.getActiveMipmapLevel(),zg=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);let l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,i,s,l,o),n>0&&this._blur(l,0,0,n),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,n=null){return this._fromTexture(e,n)}fromCubemap(e,n=null){return this._fromTexture(e,n)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=uA(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=cA(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Ng,Og,Fg),this._renderer.xr.enabled=zg,e.scissorTest=!1,Mo(e,0,0,e.width,e.height)}_fromTexture(e,n){e.mapping===xr||e.mapping===na?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Ng=this._renderer.getRenderTarget(),Og=this._renderer.getActiveCubeFace(),Fg=this._renderer.getActiveMipmapLevel(),zg=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;let i=n||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){let e=3*Math.max(this._cubeSize,112),n=4*this._cubeSize,i={magFilter:xt,minFilter:xt,generateMipmaps:!1,type:Qi,format:yi,colorSpace:Ts,depthBuffer:!1},s=lA(e,n,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==n){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=lA(e,n,i);let{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=SC(r)),this._blurMaterial=MC(r,e,n),this._ggxMaterial=AC(r,e,n)}return s}_compileMaterial(e){let n=new Mn(new xi,e);this._renderer.compile(n,rc)}_sceneToCubeUV(e,n,i,s,r){let l=new Sn(90,1,n,i),c=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],p=this._renderer,u=p.autoClear,d=p.toneMapping;p.getClearColor(oA),p.toneMapping=Di,p.autoClear=!1,p.state.buffers.depth.getReversed()&&(p.setRenderTarget(s),p.clearDepth(),p.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new Mn(new _o,new Hl({name:"PMREM.Background",side:Jt,depthWrite:!1,depthTest:!1})));let M=this._backgroundBox,m=M.material,f=!1,g=e.background;g?g.isColor&&(m.color.copy(g),e.background=null,f=!0):(m.color.copy(oA),f=!0);for(let S=0;S<6;S++){let _=S%3;_===0?(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+h[S],r.y,r.z)):_===1?(l.up.set(0,0,c[S]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+h[S],r.z)):(l.up.set(0,c[S],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+h[S]));let T=this._cubeSize;Mo(s,_*T,S>2?T:0,T,T),p.setRenderTarget(s),f&&p.render(M,l),p.render(e,l)}p.toneMapping=d,p.autoClear=u,e.background=g}_textureToCubeUV(e,n){let i=this._renderer,s=e.mapping===xr||e.mapping===na;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=uA()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=cA());let r=s?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=r;let o=r.uniforms;o.envMap.value=e;let l=this._cubeSize;Mo(n,0,0,3*l,2*l),i.setRenderTarget(n),i.render(a,rc)}_applyPMREM(e){let n=this._renderer,i=n.autoClear;n.autoClear=!1;let s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);n.autoClear=i}_applyGGXFilter(e,n,i){let s=this._renderer,r=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[i];o.material=a;let l=a.uniforms,c=i/(this._lodMeshes.length-1),h=n/(this._lodMeshes.length-1),p=Math.sqrt(c*c-h*h),u=0+c*1.25,d=p*u,{_lodMax:v}=this,M=this._sizeLods[i],m=3*M*(i>v-Ar?i-v+Ar:0),f=4*(this._cubeSize-M);l.envMap.value=e.texture,l.roughness.value=d,l.mipInt.value=v-n,Mo(r,m,f,3*M,2*M),s.setRenderTarget(r),s.render(o,rc),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=v-i,Mo(e,m,f,3*M,2*M),s.setRenderTarget(e),s.render(o,rc)}_blur(e,n,i,s,r){let a=this._pingPongRenderTarget;this._halfBlur(e,a,n,i,s,"latitudinal",r),this._halfBlur(a,e,i,i,s,"longitudinal",r)}_halfBlur(e,n,i,s,r,a,o){let l=this._renderer,c=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&Ue("blur direction must be either latitudinal or longitudinal!");let h=3,p=this._lodMeshes[s];p.material=c;let u=c.uniforms,d=this._sizeLods[i]-1,v=isFinite(r)?Math.PI/(2*d):2*Math.PI/(2*sa-1),M=r/v,m=isFinite(r)?1+Math.floor(h*M):sa;m>sa&&Re(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${sa}`);let f=[],g=0;for(let w=0;w<sa;++w){let x=w/M,E=Math.exp(-x*x/2);f.push(E),w===0?g+=E:w<m&&(g+=2*E)}for(let w=0;w<f.length;w++)f[w]=f[w]/g;u.envMap.value=e.texture,u.samples.value=m,u.weights.value=f,u.latitudinal.value=a==="latitudinal",o&&(u.poleAxis.value=o);let{_lodMax:S}=this;u.dTheta.value=v,u.mipInt.value=S-i;let _=this._sizeLods[s],T=3*_*(s>S-Ar?s-S+Ar:0),b=4*(this._cubeSize-_);Mo(n,T,b,3*_,2*_),l.setRenderTarget(n),l.render(p,rc)}};function SC(t){let e=[],n=[],i=[],s=t,r=t-Ar+1+aA.length;for(let a=0;a<r;a++){let o=Math.pow(2,s);e.push(o);let l=1/o;a>t-Ar?l=aA[a-t+Ar-1]:a===0&&(l=0),n.push(l);let c=1/(o-2),h=-c,p=1+c,u=[h,h,p,h,p,p,h,h,p,p,h,p],d=6,v=6,M=3,m=2,f=1,g=new Float32Array(M*v*d),S=new Float32Array(m*v*d),_=new Float32Array(f*v*d);for(let b=0;b<d;b++){let w=b%3*2/3-1,x=b>2?0:-1,E=[w,x,0,w+2/3,x,0,w+2/3,x+1,0,w,x,0,w+2/3,x+1,0,w,x+1,0];g.set(E,M*v*b),S.set(u,m*v*b);let R=[b,b,b,b,b,b];_.set(R,f*v*b)}let T=new xi;T.setAttribute("position",new An(g,M)),T.setAttribute("uv",new An(S,m)),T.setAttribute("faceIndex",new An(_,f)),i.push(new Mn(T,null)),s>Ar&&s--}return{lodMeshes:i,sizeLods:e,sigmas:n}}function lA(t,e,n){let i=new Nt(t,e,n);return i.texture.mapping=Zl,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function Mo(t,e,n,i,s){t.viewport.set(e,n,i,s),t.scissor.set(e,n,i,s)}function AC(t,e,n){return new Vt({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:yC,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:Nh(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:Gn,depthTest:!1,depthWrite:!1})}function MC(t,e,n){let i=new Float32Array(sa),s=new z(0,1,0);return new Vt({name:"SphericalGaussianBlur",defines:{n:sa,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Nh(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Gn,depthTest:!1,depthWrite:!1})}function cA(){return new Vt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Nh(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Gn,depthTest:!1,depthWrite:!1})}function uA(){return new Vt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Nh(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Gn,depthTest:!1,depthWrite:!1})}function Nh(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}var Ih=class extends Nt{constructor(e=1,n={}){super(e,e,n),this.isWebGLCubeRenderTarget=!0;let i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new kl(s),this._setTextureOptions(n),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,n){this.texture.type=n.type,this.texture.colorSpace=n.colorSpace,this.texture.generateMipmaps=n.generateMipmaps,this.texture.minFilter=n.minFilter,this.texture.magFilter=n.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new _o(5,5,5),r=new Vt({name:"CubemapFromEquirect",uniforms:ia(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:Jt,blending:Gn});r.uniforms.tEquirect.value=n;let a=new Mn(s,r),o=n.minFilter;return n.minFilter===yr&&(n.minFilter=xt),new Gf(1,10,this).update(e,a),n.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,n=!0,i=!0,s=!0){let r=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(n,i,s);e.setRenderTarget(r)}};function EC(t){let e=new WeakMap,n=new WeakMap,i=null;function s(u,d=!1){return u==null?null:d?a(u):r(u)}function r(u){if(u&&u.isTexture){let d=u.mapping;if(d===kf||d===Wf)if(e.has(u)){let v=e.get(u).texture;return o(v,u.mapping)}else{let v=u.image;if(v&&v.height>0){let M=new Ih(v.height);return M.fromEquirectangularTexture(t,u),e.set(u,M),u.addEventListener("dispose",c),o(M.texture,u.mapping)}else return null}}return u}function a(u){if(u&&u.isTexture){let d=u.mapping,v=d===kf||d===Wf,M=d===xr||d===na;if(v||M){let m=n.get(u),f=m!==void 0?m.texture.pmremVersion:0;if(u.isRenderTargetTexture&&u.pmremVersion!==f)return i===null&&(i=new Bh(t)),m=v?i.fromEquirectangular(u,m):i.fromCubemap(u,m),m.texture.pmremVersion=u.pmremVersion,n.set(u,m),m.texture;if(m!==void 0)return m.texture;{let g=u.image;return v&&g&&g.height>0||M&&g&&l(g)?(i===null&&(i=new Bh(t)),m=v?i.fromEquirectangular(u):i.fromCubemap(u),m.texture.pmremVersion=u.pmremVersion,n.set(u,m),u.addEventListener("dispose",h),m.texture):null}}}return u}function o(u,d){return d===kf?u.mapping=xr:d===Wf&&(u.mapping=na),u}function l(u){let d=0,v=6;for(let M=0;M<v;M++)u[M]!==void 0&&d++;return d===v}function c(u){let d=u.target;d.removeEventListener("dispose",c);let v=e.get(d);v!==void 0&&(e.delete(d),v.dispose())}function h(u){let d=u.target;d.removeEventListener("dispose",h);let v=n.get(d);v!==void 0&&(n.delete(d),v.dispose())}function p(){e=new WeakMap,n=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:s,dispose:p}}function TC(t){let e={};function n(i){if(e[i]!==void 0)return e[i];let s=t.getExtension(i);return e[i]=s,s}return{has:function(i){return n(i)!==null},init:function(){n("EXT_color_buffer_float"),n("WEBGL_clip_cull_distance"),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float"),n("WEBGL_multisampled_render_to_texture"),n("WEBGL_render_shared_exponent")},get:function(i){let s=n(i);return s===null&&jr("WebGLRenderer: "+i+" extension not supported."),s}}}function bC(t,e,n,i){let s={},r=new WeakMap;function a(p){let u=p.target;u.index!==null&&e.remove(u.index);for(let v in u.attributes)e.remove(u.attributes[v]);u.removeEventListener("dispose",a),delete s[u.id];let d=r.get(u);d&&(e.remove(d),r.delete(u)),i.releaseStatesOfGeometry(u),u.isInstancedBufferGeometry===!0&&delete u._maxInstanceCount,n.memory.geometries--}function o(p,u){return s[u.id]===!0||(u.addEventListener("dispose",a),s[u.id]=!0,n.memory.geometries++),u}function l(p){let u=p.attributes;for(let d in u)e.update(u[d],t.ARRAY_BUFFER)}function c(p){let u=[],d=p.index,v=p.attributes.position,M=0;if(v===void 0)return;if(d!==null){let g=d.array;M=d.version;for(let S=0,_=g.length;S<_;S+=3){let T=g[S+0],b=g[S+1],w=g[S+2];u.push(T,b,b,w,w,T)}}else{let g=v.array;M=v.version;for(let S=0,_=g.length/3-1;S<_;S+=3){let T=S+0,b=S+1,w=S+2;u.push(T,b,b,w,w,T)}}let m=new(v.count>=65535?Gl:zl)(u,1);m.version=M;let f=r.get(p);f&&e.remove(f),r.set(p,m)}function h(p){let u=r.get(p);if(u){let d=p.index;d!==null&&u.version<d.version&&c(p)}else c(p);return r.get(p)}return{get:o,update:l,getWireframeAttribute:h}}function wC(t,e,n){let i;function s(p){i=p}let r,a;function o(p){r=p.type,a=p.bytesPerElement}function l(p,u){t.drawElements(i,u,r,p*a),n.update(u,i,1)}function c(p,u,d){d!==0&&(t.drawElementsInstanced(i,u,r,p*a,d),n.update(u,i,d))}function h(p,u,d){if(d===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,u,0,r,p,0,d);let M=0;for(let m=0;m<d;m++)M+=u[m];n.update(M,i,1)}this.setMode=s,this.setIndex=o,this.render=l,this.renderInstances=c,this.renderMultiDraw=h}function CC(t){let e={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,a,o){switch(n.calls++,a){case t.TRIANGLES:n.triangles+=o*(r/3);break;case t.LINES:n.lines+=o*(r/2);break;case t.LINE_STRIP:n.lines+=o*(r-1);break;case t.LINE_LOOP:n.lines+=o*r;break;case t.POINTS:n.points+=o*r;break;default:Ue("WebGLInfo: Unknown draw mode:",a);break}}function s(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:e,render:n,programs:null,autoReset:!0,reset:s,update:i}}function RC(t,e,n){let i=new WeakMap,s=new Dt;function r(a,o,l){let c=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,p=h!==void 0?h.length:0,u=i.get(o);if(u===void 0||u.count!==p){let E=function(){w.dispose(),i.delete(o),o.removeEventListener("dispose",E)};u!==void 0&&u.texture.dispose();let d=o.morphAttributes.position!==void 0,v=o.morphAttributes.normal!==void 0,M=o.morphAttributes.color!==void 0,m=o.morphAttributes.position||[],f=o.morphAttributes.normal||[],g=o.morphAttributes.color||[],S=0;d===!0&&(S=1),v===!0&&(S=2),M===!0&&(S=3);let _=o.attributes.position.count*S,T=1;_>e.maxTextureSize&&(T=Math.ceil(_/e.maxTextureSize),_=e.maxTextureSize);let b=new Float32Array(_*T*4*p),w=new Ol(b,_,T,p);w.type=ti,w.needsUpdate=!0;let x=S*4;for(let R=0;R<p;R++){let D=m[R],L=f[R],q=g[R],Y=_*T*4*R;for(let N=0;N<D.count;N++){let k=N*x;d===!0&&(s.fromBufferAttribute(D,N),b[Y+k+0]=s.x,b[Y+k+1]=s.y,b[Y+k+2]=s.z,b[Y+k+3]=0),v===!0&&(s.fromBufferAttribute(L,N),b[Y+k+4]=s.x,b[Y+k+5]=s.y,b[Y+k+6]=s.z,b[Y+k+7]=0),M===!0&&(s.fromBufferAttribute(q,N),b[Y+k+8]=s.x,b[Y+k+9]=s.y,b[Y+k+10]=s.z,b[Y+k+11]=q.itemSize===4?s.w:1)}}u={count:p,texture:w,size:new Ie(_,T)},i.set(o,u),o.addEventListener("dispose",E)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)l.getUniforms().setValue(t,"morphTexture",a.morphTexture,n);else{let d=0;for(let M=0;M<c.length;M++)d+=c[M];let v=o.morphTargetsRelative?1:1-d;l.getUniforms().setValue(t,"morphTargetBaseInfluence",v),l.getUniforms().setValue(t,"morphTargetInfluences",c)}l.getUniforms().setValue(t,"morphTargetsTexture",u.texture,n),l.getUniforms().setValue(t,"morphTargetsTextureSize",u.size)}return{update:r}}function DC(t,e,n,i,s){let r=new WeakMap;function a(c){let h=s.render.frame,p=c.geometry,u=e.get(c,p);if(r.get(u)!==h&&(e.update(u),r.set(u,h)),c.isInstancedMesh&&(c.hasEventListener("dispose",l)===!1&&c.addEventListener("dispose",l),r.get(c)!==h&&(n.update(c.instanceMatrix,t.ARRAY_BUFFER),c.instanceColor!==null&&n.update(c.instanceColor,t.ARRAY_BUFFER),r.set(c,h))),c.isSkinnedMesh){let d=c.skeleton;r.get(d)!==h&&(d.update(),r.set(d,h))}return u}function o(){r=new WeakMap}function l(c){let h=c.target;h.removeEventListener("dispose",l),i.releaseStatesOfObject(h),n.remove(h.instanceMatrix),h.instanceColor!==null&&n.remove(h.instanceColor)}return{update:a,dispose:o}}var UC={[ug]:"LINEAR_TONE_MAPPING",[fg]:"REINHARD_TONE_MAPPING",[hg]:"CINEON_TONE_MAPPING",[dg]:"ACES_FILMIC_TONE_MAPPING",[mg]:"AGX_TONE_MAPPING",[gg]:"NEUTRAL_TONE_MAPPING",[pg]:"CUSTOM_TONE_MAPPING"};function BC(t,e,n,i,s,r){let a=new Nt(e,n,{type:t,depthBuffer:s,stencilBuffer:r,samples:i?4:0,depthTexture:s?new Ri(e,n):void 0}),o=new Nt(e,n,{type:Qi,depthBuffer:!1,stencilBuffer:!1}),l=new xi;l.setAttribute("position",new gi([-1,3,0,-1,-1,0,3,-1,0],3)),l.setAttribute("uv",new gi([0,2,0,0,2,0],2));let c=new wf({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),h=new Mn(l,c),p=new bs(-1,1,1,-1,0,1),u=null,d=null,v=!1,M,m=null,f=[],g=!1;this.setSize=function(S,_){a.setSize(S,_),o.setSize(S,_);for(let T=0;T<f.length;T++){let b=f[T];b.setSize&&b.setSize(S,_)}},this.setEffects=function(S){f=S,g=f.length>0&&f[0].isRenderPass===!0;let _=a.width,T=a.height;for(let b=0;b<f.length;b++){let w=f[b];w.setSize&&w.setSize(_,T)}},this.begin=function(S,_){if(v||S.toneMapping===Di&&f.length===0)return!1;if(m=_,_!==null){let T=_.width,b=_.height;(a.width!==T||a.height!==b)&&this.setSize(T,b)}return g===!1&&S.setRenderTarget(a),M=S.toneMapping,S.toneMapping=Di,!0},this.hasRenderPass=function(){return g},this.end=function(S,_){S.toneMapping=M,v=!0;let T=a,b=o;for(let w=0;w<f.length;w++){let x=f[w];if(x.enabled!==!1&&(x.render(S,b,T,_),x.needsSwap!==!1)){let E=T;T=b,b=E}}if(u!==S.outputColorSpace||d!==S.toneMapping){u=S.outputColorSpace,d=S.toneMapping,c.defines={},Ye.getTransfer(u)===st&&(c.defines.SRGB_TRANSFER="");let w=UC[d];w&&(c.defines[w]=""),c.needsUpdate=!0}c.uniforms.tDiffuse.value=T.texture,S.setRenderTarget(m),S.render(h,p),m=null,v=!1},this.isCompositing=function(){return v},this.dispose=function(){a.depthTexture&&a.depthTexture.dispose(),a.dispose(),o.dispose(),l.dispose(),c.dispose()}}var DA=new Kt,Vg=new Ri(1,1),UA=new Ol,BA=new Mf,IA=new kl,fA=[],hA=[],dA=new Float32Array(16),pA=new Float32Array(9),mA=new Float32Array(4);function To(t,e,n){let i=t[0];if(i<=0||i>0)return t;let s=e*n,r=fA[s];if(r===void 0&&(r=new Float32Array(s),fA[s]=r),e!==0){i.toArray(r,0);for(let a=1,o=0;a!==e;++a)o+=n,t[a].toArray(r,o)}return r}function nn(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function sn(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function Oh(t,e){let n=hA[e];n===void 0&&(n=new Int32Array(e),hA[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function IC(t,e){let n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function PC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nn(n,e))return;t.uniform2fv(this.addr,e),sn(n,e)}}function LC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(e.r!==void 0)(n[0]!==e.r||n[1]!==e.g||n[2]!==e.b)&&(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(nn(n,e))return;t.uniform3fv(this.addr,e),sn(n,e)}}function NC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nn(n,e))return;t.uniform4fv(this.addr,e),sn(n,e)}}function OC(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(nn(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),sn(n,e)}else{if(nn(n,i))return;mA.set(i),t.uniformMatrix2fv(this.addr,!1,mA),sn(n,i)}}function FC(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(nn(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),sn(n,e)}else{if(nn(n,i))return;pA.set(i),t.uniformMatrix3fv(this.addr,!1,pA),sn(n,i)}}function zC(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(nn(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),sn(n,e)}else{if(nn(n,i))return;dA.set(i),t.uniformMatrix4fv(this.addr,!1,dA),sn(n,i)}}function GC(t,e){let n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function HC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2i(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nn(n,e))return;t.uniform2iv(this.addr,e),sn(n,e)}}function VC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3i(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(nn(n,e))return;t.uniform3iv(this.addr,e),sn(n,e)}}function kC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4i(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nn(n,e))return;t.uniform4iv(this.addr,e),sn(n,e)}}function WC(t,e){let n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function XC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2ui(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nn(n,e))return;t.uniform2uiv(this.addr,e),sn(n,e)}}function YC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3ui(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(nn(n,e))return;t.uniform3uiv(this.addr,e),sn(n,e)}}function qC(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4ui(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nn(n,e))return;t.uniform4uiv(this.addr,e),sn(n,e)}}function QC(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s);let r;this.type===t.SAMPLER_2D_SHADOW?(Vg.compareFunction=n.isReversedDepthBuffer()?Rh:Ch,r=Vg):r=DA,n.setTexture2D(e||r,s)}function ZC(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture3D(e||BA,s)}function KC(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTextureCube(e||IA,s)}function JC(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture2DArray(e||UA,s)}function jC(t){switch(t){case 5126:return IC;case 35664:return PC;case 35665:return LC;case 35666:return NC;case 35674:return OC;case 35675:return FC;case 35676:return zC;case 5124:case 35670:return GC;case 35667:case 35671:return HC;case 35668:case 35672:return VC;case 35669:case 35673:return kC;case 5125:return WC;case 36294:return XC;case 36295:return YC;case 36296:return qC;case 35678:case 36198:case 36298:case 36306:case 35682:return QC;case 35679:case 36299:case 36307:return ZC;case 35680:case 36300:case 36308:case 36293:return KC;case 36289:case 36303:case 36311:case 36292:return JC}}function $C(t,e){t.uniform1fv(this.addr,e)}function e2(t,e){let n=To(e,this.size,2);t.uniform2fv(this.addr,n)}function t2(t,e){let n=To(e,this.size,3);t.uniform3fv(this.addr,n)}function n2(t,e){let n=To(e,this.size,4);t.uniform4fv(this.addr,n)}function i2(t,e){let n=To(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function s2(t,e){let n=To(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function r2(t,e){let n=To(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function a2(t,e){t.uniform1iv(this.addr,e)}function o2(t,e){t.uniform2iv(this.addr,e)}function l2(t,e){t.uniform3iv(this.addr,e)}function c2(t,e){t.uniform4iv(this.addr,e)}function u2(t,e){t.uniform1uiv(this.addr,e)}function f2(t,e){t.uniform2uiv(this.addr,e)}function h2(t,e){t.uniform3uiv(this.addr,e)}function d2(t,e){t.uniform4uiv(this.addr,e)}function p2(t,e,n){let i=this.cache,s=e.length,r=Oh(n,s);nn(i,r)||(t.uniform1iv(this.addr,r),sn(i,r));let a;this.type===t.SAMPLER_2D_SHADOW?a=Vg:a=DA;for(let o=0;o!==s;++o)n.setTexture2D(e[o]||a,r[o])}function m2(t,e,n){let i=this.cache,s=e.length,r=Oh(n,s);nn(i,r)||(t.uniform1iv(this.addr,r),sn(i,r));for(let a=0;a!==s;++a)n.setTexture3D(e[a]||BA,r[a])}function g2(t,e,n){let i=this.cache,s=e.length,r=Oh(n,s);nn(i,r)||(t.uniform1iv(this.addr,r),sn(i,r));for(let a=0;a!==s;++a)n.setTextureCube(e[a]||IA,r[a])}function v2(t,e,n){let i=this.cache,s=e.length,r=Oh(n,s);nn(i,r)||(t.uniform1iv(this.addr,r),sn(i,r));for(let a=0;a!==s;++a)n.setTexture2DArray(e[a]||UA,r[a])}function x2(t){switch(t){case 5126:return $C;case 35664:return e2;case 35665:return t2;case 35666:return n2;case 35674:return i2;case 35675:return s2;case 35676:return r2;case 5124:case 35670:return a2;case 35667:case 35671:return o2;case 35668:case 35672:return l2;case 35669:case 35673:return c2;case 5125:return u2;case 36294:return f2;case 36295:return h2;case 36296:return d2;case 35678:case 36198:case 36298:case 36306:case 35682:return p2;case 35679:case 36299:case 36307:return m2;case 35680:case 36300:case 36308:case 36293:return g2;case 36289:case 36303:case 36311:case 36292:return v2}}var kg=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.setValue=jC(n.type)}},Wg=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.size=n.size,this.setValue=x2(n.type)}},Xg=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,n,i){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(e,n[o.id],i)}}},Gg=/(\w+)(\])?(\[|\.)?/g;function gA(t,e){t.seq.push(e),t.map[e.id]=e}function y2(t,e,n){let i=t.name,s=i.length;for(Gg.lastIndex=0;;){let r=Gg.exec(i),a=Gg.lastIndex,o=r[1],l=r[2]==="]",c=r[3];if(l&&(o=o|0),c===void 0||c==="["&&a+2===s){gA(n,c===void 0?new kg(o,t,e):new Wg(o,t,e));break}else{let p=n.map[o];p===void 0&&(p=new Xg(o),gA(n,p)),n=p}}}var Eo=class{constructor(e,n){this.seq=[],this.map={};let i=e.getProgramParameter(n,e.ACTIVE_UNIFORMS);for(let a=0;a<i;++a){let o=e.getActiveUniform(n,a),l=e.getUniformLocation(n,o.name);y2(o,l,this)}let s=[],r=[];for(let a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(e,n,i,s){let r=this.map[n];r!==void 0&&r.setValue(e,i,s)}setOptional(e,n,i){let s=n[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,n,i,s){for(let r=0,a=n.length;r!==a;++r){let o=n[r],l=i[o.id];l.needsUpdate!==!1&&o.setValue(e,l.value,s)}}static seqWithValue(e,n){let i=[];for(let s=0,r=e.length;s!==r;++s){let a=e[s];a.id in n&&i.push(a)}return i}};function vA(t,e,n){let i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}var _2=37297,S2=0;function A2(t,e){let n=t.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,n.length);for(let a=s;a<r;a++){let o=a+1;i.push(`${o===e?">":" "} ${o}: ${n[a]}`)}return i.join(`
`)}var xA=new Pe;function M2(t){Ye._getMatrix(xA,Ye.workingColorSpace,t);let e=`mat3( ${xA.elements.map(n=>n.toFixed(4))} )`;switch(Ye.getTransfer(t)){case Pl:return[e,"LinearTransferOETF"];case st:return[e,"sRGBTransferOETF"];default:return Re("WebGLProgram: Unsupported color space: ",t),[e,"LinearTransferOETF"]}}function yA(t,e,n){let i=t.getShaderParameter(e,t.COMPILE_STATUS),r=(t.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return n.toUpperCase()+`

`+r+`

`+A2(t.getShaderSource(e),o)}else return r}function E2(t,e){let n=M2(e);return[`vec4 ${t}( vec4 value ) {`,`	return ${n[1]}( vec4( value.rgb * ${n[0]}, value.a ) );`,"}"].join(`
`)}var T2={[ug]:"Linear",[fg]:"Reinhard",[hg]:"Cineon",[dg]:"ACESFilmic",[mg]:"AgX",[gg]:"Neutral",[pg]:"Custom"};function b2(t,e){let n=T2[e];return n===void 0?(Re("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+t+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}var Uh=new z;function w2(){Ye.getLuminanceCoefficients(Uh);let t=Uh.x.toFixed(4),e=Uh.y.toFixed(4),n=Uh.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${t}, ${e}, ${n} );`,"	return dot( weights, rgb );","}"].join(`
`)}function C2(t){return[t.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",t.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(oc).join(`
`)}function R2(t){let e=[];for(let n in t){let i=t[n];i!==!1&&e.push("#define "+n+" "+i)}return e.join(`
`)}function D2(t,e){let n={},i=t.getProgramParameter(e,t.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=t.getActiveAttrib(e,s),a=r.name,o=1;r.type===t.FLOAT_MAT2&&(o=2),r.type===t.FLOAT_MAT3&&(o=3),r.type===t.FLOAT_MAT4&&(o=4),n[a]={type:r.type,location:t.getAttribLocation(e,a),locationSize:o}}return n}function oc(t){return t!==""}function _A(t,e){let n=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return t.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function SA(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var U2=/^[ \t]*#include +<([\w\d./]+)>/gm;function Yg(t){return t.replace(U2,I2)}var B2=new Map;function I2(t,e){let n=ze[e];if(n===void 0){let i=B2.get(e);if(i!==void 0)n=ze[i],Re('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">")}return Yg(n)}var P2=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function AA(t){return t.replace(P2,L2)}function L2(t,e,n,i){let s="";for(let r=parseInt(e);r<parseInt(n);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function MA(t){let e=`precision ${t.precision} float;
	precision ${t.precision} int;
	precision ${t.precision} sampler2D;
	precision ${t.precision} samplerCube;
	precision ${t.precision} sampler3D;
	precision ${t.precision} sampler2DArray;
	precision ${t.precision} sampler2DShadow;
	precision ${t.precision} samplerCubeShadow;
	precision ${t.precision} sampler2DArrayShadow;
	precision ${t.precision} isampler2D;
	precision ${t.precision} isampler3D;
	precision ${t.precision} isamplerCube;
	precision ${t.precision} isampler2DArray;
	precision ${t.precision} usampler2D;
	precision ${t.precision} usampler3D;
	precision ${t.precision} usamplerCube;
	precision ${t.precision} usampler2DArray;
	`;return t.precision==="highp"?e+=`
#define HIGH_PRECISION`:t.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:t.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}var N2={[Ql]:"SHADOWMAP_TYPE_PCF",[So]:"SHADOWMAP_TYPE_VSM"};function O2(t){return N2[t.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var F2={[xr]:"ENVMAP_TYPE_CUBE",[na]:"ENVMAP_TYPE_CUBE",[Zl]:"ENVMAP_TYPE_CUBE_UV"};function z2(t){return t.envMap===!1?"ENVMAP_TYPE_CUBE":F2[t.envMapMode]||"ENVMAP_TYPE_CUBE"}var G2={[na]:"ENVMAP_MODE_REFRACTION"};function H2(t){return t.envMap===!1?"ENVMAP_MODE_REFLECTION":G2[t.envMapMode]||"ENVMAP_MODE_REFLECTION"}var V2={[cg]:"ENVMAP_BLENDING_MULTIPLY",[kS]:"ENVMAP_BLENDING_MIX",[WS]:"ENVMAP_BLENDING_ADD"};function k2(t){return t.envMap===!1?"ENVMAP_BLENDING_NONE":V2[t.combine]||"ENVMAP_BLENDING_NONE"}function W2(t){let e=t.envMapCubeUVHeight;if(e===null)return null;let n=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,n),112)),texelHeight:i,maxMip:n}}function X2(t,e,n,i){let s=t.getContext(),r=n.defines,a=n.vertexShader,o=n.fragmentShader,l=O2(n),c=z2(n),h=H2(n),p=k2(n),u=W2(n),d=C2(n),v=R2(r),M=s.createProgram(),m,f,g=n.glslVersion?"#version "+n.glslVersion+`
`:"";n.isRawShaderMaterial?(m=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v].filter(oc).join(`
`),m.length>0&&(m+=`
`),f=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v].filter(oc).join(`
`),f.length>0&&(f+=`
`)):(m=[MA(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v,n.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",n.batching?"#define USE_BATCHING":"",n.batchingColor?"#define USE_BATCHING_COLOR":"",n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.instancingMorph?"#define USE_INSTANCING_MORPH":"",n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+h:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.displacementMap?"#define USE_DISPLACEMENTMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.mapUv?"#define MAP_UV "+n.mapUv:"",n.alphaMapUv?"#define ALPHAMAP_UV "+n.alphaMapUv:"",n.lightMapUv?"#define LIGHTMAP_UV "+n.lightMapUv:"",n.aoMapUv?"#define AOMAP_UV "+n.aoMapUv:"",n.emissiveMapUv?"#define EMISSIVEMAP_UV "+n.emissiveMapUv:"",n.bumpMapUv?"#define BUMPMAP_UV "+n.bumpMapUv:"",n.normalMapUv?"#define NORMALMAP_UV "+n.normalMapUv:"",n.displacementMapUv?"#define DISPLACEMENTMAP_UV "+n.displacementMapUv:"",n.metalnessMapUv?"#define METALNESSMAP_UV "+n.metalnessMapUv:"",n.roughnessMapUv?"#define ROUGHNESSMAP_UV "+n.roughnessMapUv:"",n.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+n.anisotropyMapUv:"",n.clearcoatMapUv?"#define CLEARCOATMAP_UV "+n.clearcoatMapUv:"",n.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+n.clearcoatNormalMapUv:"",n.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+n.clearcoatRoughnessMapUv:"",n.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+n.iridescenceMapUv:"",n.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+n.iridescenceThicknessMapUv:"",n.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+n.sheenColorMapUv:"",n.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+n.sheenRoughnessMapUv:"",n.specularMapUv?"#define SPECULARMAP_UV "+n.specularMapUv:"",n.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+n.specularColorMapUv:"",n.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+n.specularIntensityMapUv:"",n.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+n.transmissionMapUv:"",n.thicknessMapUv?"#define THICKNESSMAP_UV "+n.thicknessMapUv:"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexNormals?"#define HAS_NORMAL":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&n.flatShading===!1?"#define USE_MORPHNORMALS":"",n.morphColors?"#define USE_MORPHCOLORS":"",n.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+n.morphTextureStride:"",n.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(oc).join(`
`),f=[MA(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,v,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+c:"",n.envMap?"#define "+h:"",n.envMap?"#define "+p:"",u?"#define CUBEUV_TEXEL_WIDTH "+u.texelWidth:"",u?"#define CUBEUV_TEXEL_HEIGHT "+u.texelHeight:"",u?"#define CUBEUV_MAX_MIP "+u.maxMip+".0":"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.dispersion?"#define USE_DISPERSION":"",n.iridescence?"#define USE_IRIDESCENCE":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas||n.batchingColor?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.numLightProbeGrids>0?"#define USE_LIGHT_PROBES_GRID":"",n.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",n.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",n.toneMapping!==Di?"#define TONE_MAPPING":"",n.toneMapping!==Di?ze.tonemapping_pars_fragment:"",n.toneMapping!==Di?b2("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.opaque?"#define OPAQUE":"",ze.colorspace_pars_fragment,E2("linearToOutputTexel",n.outputColorSpace),w2(),n.useDepthPacking?"#define DEPTH_PACKING "+n.depthPacking:"",`
`].filter(oc).join(`
`)),a=Yg(a),a=_A(a,n),a=SA(a,n),o=Yg(o),o=_A(o,n),o=SA(o,n),a=AA(a),o=AA(o),n.isRawShaderMaterial!==!0&&(g=`#version 300 es
`,m=[d,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,f=["#define varying in",n.glslVersion===ic?"":"layout(location = 0) out highp vec4 pc_fragColor;",n.glslVersion===ic?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+f);let S=g+m+a,_=g+f+o,T=vA(s,s.VERTEX_SHADER,S),b=vA(s,s.FRAGMENT_SHADER,_);s.attachShader(M,T),s.attachShader(M,b),n.index0AttributeName!==void 0?s.bindAttribLocation(M,0,n.index0AttributeName):n.hasPositionAttribute===!0&&s.bindAttribLocation(M,0,"position"),s.linkProgram(M);function w(D){if(t.debug.checkShaderErrors){let L=s.getProgramInfoLog(M)||"",q=s.getShaderInfoLog(T)||"",Y=s.getShaderInfoLog(b)||"",N=L.trim(),k=q.trim(),V=Y.trim(),j=!0,ee=!0;if(s.getProgramParameter(M,s.LINK_STATUS)===!1)if(j=!1,typeof t.debug.onShaderError=="function")t.debug.onShaderError(s,M,T,b);else{let se=yA(s,T,"vertex"),he=yA(s,b,"fragment");Ue("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(M,s.VALIDATE_STATUS)+`

Material Name: `+D.name+`
Material Type: `+D.type+`

Program Info Log: `+N+`
`+se+`
`+he)}else N!==""?Re("WebGLProgram: Program Info Log:",N):(k===""||V==="")&&(ee=!1);ee&&(D.diagnostics={runnable:j,programLog:N,vertexShader:{log:k,prefix:m},fragmentShader:{log:V,prefix:f}})}s.deleteShader(T),s.deleteShader(b),x=new Eo(s,M),E=D2(s,M)}let x;this.getUniforms=function(){return x===void 0&&w(this),x};let E;this.getAttributes=function(){return E===void 0&&w(this),E};let R=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return R===!1&&(R=s.getProgramParameter(M,_2)),R},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(M),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=S2++,this.cacheKey=e,this.usedTimes=1,this.program=M,this.vertexShader=T,this.fragmentShader=b,this}var Y2=0,qg=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,n,i){let s=this._getShaderCacheForMaterial(e);return s.has(n)===!1&&(s.add(n),n.usedTimes++),s.has(i)===!1&&(s.add(i),i.usedTimes++),this}remove(e){let n=this.materialCache.get(e);for(let i of n)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let n=this.materialCache,i=n.get(e);return i===void 0&&(i=new Set,n.set(e,i)),i}_getShaderStage(e){let n=this.shaderCache,i=n.get(e);return i===void 0&&(i=new Qg(e),n.set(e,i)),i}},Qg=class{constructor(e){this.id=Y2++,this.code=e,this.usedTimes=0}};function q2(t){return t===Sr||t===tc||t===nc}function Q2(t,e,n,i,s,r){let a=new Fl,o=new qg,l=new Set,c=[],h=new Map,p=i.logarithmicDepthBuffer,u=i.precision,d={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function v(x){return l.add(x),x===0?"uv":`uv${x}`}function M(x,E,R,D,L,q){let Y=D.fog,N=L.geometry,k=x.isMeshStandardMaterial||x.isMeshLambertMaterial||x.isMeshPhongMaterial?D.environment:null,V=x.isMeshStandardMaterial||x.isMeshLambertMaterial&&!x.envMap||x.isMeshPhongMaterial&&!x.envMap,j=e.get(x.envMap||k,V),ee=j&&j.mapping===Zl?j.image.height:null,se=d[x.type];x.precision!==null&&(u=i.getMaxPrecision(x.precision),u!==x.precision&&Re("WebGLProgram.getParameters:",x.precision,"not supported, using",u,"instead."));let he=N.morphAttributes.position||N.morphAttributes.normal||N.morphAttributes.color,ve=he!==void 0?he.length:0,Ke=0;N.morphAttributes.position!==void 0&&(Ke=1),N.morphAttributes.normal!==void 0&&(Ke=2),N.morphAttributes.color!==void 0&&(Ke=3);let yt,Je,Z,ie;if(se){let ye=ji[se];yt=ye.vertexShader,Je=ye.fragmentShader}else{yt=x.vertexShader,Je=x.fragmentShader;let ye=o.getVertexShaderStage(x),Bt=o.getFragmentShaderStage(x);o.update(x,ye,Bt),Z=ye.id,ie=Bt.id}let te=t.getRenderTarget(),Se=t.state.buffers.depth.getReversed(),Ae=L.isInstancedMesh===!0,Ce=L.isBatchedMesh===!0,Rt=!!x.map,We=!!x.matcap,ft=!!j,tt=!!x.aoMap,je=!!x.lightMap,kt=!!x.bumpMap&&x.wireframe===!1,$t=!!x.normalMap,rn=!!x.displacementMap,fn=!!x.emissiveMap,Ut=!!x.metalnessMap,Wt=!!x.roughnessMap,B=x.anisotropy>0,Dn=x.clearcoat>0,rt=x.dispersion>0,C=x.iridescence>0,y=x.sheen>0,P=x.transmission>0,G=B&&!!x.anisotropyMap,W=Dn&&!!x.clearcoatMap,ne=Dn&&!!x.clearcoatNormalMap,ae=Dn&&!!x.clearcoatRoughnessMap,X=C&&!!x.iridescenceMap,K=C&&!!x.iridescenceThicknessMap,oe=y&&!!x.sheenColorMap,Ee=y&&!!x.sheenRoughnessMap,ue=!!x.specularMap,le=!!x.specularColorMap,we=!!x.specularIntensityMap,De=P&&!!x.transmissionMap,Ne=P&&!!x.thicknessMap,U=!!x.gradientMap,re=!!x.alphaMap,Q=x.alphaTest>0,ce=!!x.alphaHash,me=!!x.extensions,$=Di;x.toneMapped&&(te===null||te.isXRRenderTarget===!0)&&($=t.toneMapping);let Me={shaderID:se,shaderType:x.type,shaderName:x.name,vertexShader:yt,fragmentShader:Je,defines:x.defines,customVertexShaderID:Z,customFragmentShaderID:ie,isRawShaderMaterial:x.isRawShaderMaterial===!0,glslVersion:x.glslVersion,precision:u,batching:Ce,batchingColor:Ce&&L._colorsTexture!==null,instancing:Ae,instancingColor:Ae&&L.instanceColor!==null,instancingMorph:Ae&&L.morphTexture!==null,outputColorSpace:te===null?t.outputColorSpace:te.isXRRenderTarget===!0?te.texture.colorSpace:Ye.workingColorSpace,alphaToCoverage:!!x.alphaToCoverage,map:Rt,matcap:We,envMap:ft,envMapMode:ft&&j.mapping,envMapCubeUVHeight:ee,aoMap:tt,lightMap:je,bumpMap:kt,normalMap:$t,displacementMap:rn,emissiveMap:fn,normalMapObjectSpace:$t&&x.normalMapType===YS,normalMapTangentSpace:$t&&x.normalMapType===Tg,packedNormalMap:$t&&x.normalMapType===Tg&&q2(x.normalMap.format),metalnessMap:Ut,roughnessMap:Wt,anisotropy:B,anisotropyMap:G,clearcoat:Dn,clearcoatMap:W,clearcoatNormalMap:ne,clearcoatRoughnessMap:ae,dispersion:rt,iridescence:C,iridescenceMap:X,iridescenceThicknessMap:K,sheen:y,sheenColorMap:oe,sheenRoughnessMap:Ee,specularMap:ue,specularColorMap:le,specularIntensityMap:we,transmission:P,transmissionMap:De,thicknessMap:Ne,gradientMap:U,opaque:x.transparent===!1&&x.blending===$r&&x.alphaToCoverage===!1,alphaMap:re,alphaTest:Q,alphaHash:ce,combine:x.combine,mapUv:Rt&&v(x.map.channel),aoMapUv:tt&&v(x.aoMap.channel),lightMapUv:je&&v(x.lightMap.channel),bumpMapUv:kt&&v(x.bumpMap.channel),normalMapUv:$t&&v(x.normalMap.channel),displacementMapUv:rn&&v(x.displacementMap.channel),emissiveMapUv:fn&&v(x.emissiveMap.channel),metalnessMapUv:Ut&&v(x.metalnessMap.channel),roughnessMapUv:Wt&&v(x.roughnessMap.channel),anisotropyMapUv:G&&v(x.anisotropyMap.channel),clearcoatMapUv:W&&v(x.clearcoatMap.channel),clearcoatNormalMapUv:ne&&v(x.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:ae&&v(x.clearcoatRoughnessMap.channel),iridescenceMapUv:X&&v(x.iridescenceMap.channel),iridescenceThicknessMapUv:K&&v(x.iridescenceThicknessMap.channel),sheenColorMapUv:oe&&v(x.sheenColorMap.channel),sheenRoughnessMapUv:Ee&&v(x.sheenRoughnessMap.channel),specularMapUv:ue&&v(x.specularMap.channel),specularColorMapUv:le&&v(x.specularColorMap.channel),specularIntensityMapUv:we&&v(x.specularIntensityMap.channel),transmissionMapUv:De&&v(x.transmissionMap.channel),thicknessMapUv:Ne&&v(x.thicknessMap.channel),alphaMapUv:re&&v(x.alphaMap.channel),vertexTangents:!!N.attributes.tangent&&($t||B),vertexNormals:!!N.attributes.normal,vertexColors:x.vertexColors,vertexAlphas:x.vertexColors===!0&&!!N.attributes.color&&N.attributes.color.itemSize===4,pointsUvs:L.isPoints===!0&&!!N.attributes.uv&&(Rt||re),fog:!!Y,useFog:x.fog===!0,fogExp2:!!Y&&Y.isFogExp2,flatShading:x.wireframe===!1&&(x.flatShading===!0||N.attributes.normal===void 0&&$t===!1&&(x.isMeshLambertMaterial||x.isMeshPhongMaterial||x.isMeshStandardMaterial||x.isMeshPhysicalMaterial)),sizeAttenuation:x.sizeAttenuation===!0,logarithmicDepthBuffer:p,reversedDepthBuffer:Se,skinning:L.isSkinnedMesh===!0,hasPositionAttribute:N.attributes.position!==void 0,morphTargets:N.morphAttributes.position!==void 0,morphNormals:N.morphAttributes.normal!==void 0,morphColors:N.morphAttributes.color!==void 0,morphTargetsCount:ve,morphTextureStride:Ke,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numLightProbeGrids:q.length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:x.dithering,shadowMapEnabled:t.shadowMap.enabled&&R.length>0,shadowMapType:t.shadowMap.type,toneMapping:$,decodeVideoTexture:Rt&&x.map.isVideoTexture===!0&&Ye.getTransfer(x.map.colorSpace)===st,decodeVideoTextureEmissive:fn&&x.emissiveMap.isVideoTexture===!0&&Ye.getTransfer(x.emissiveMap.colorSpace)===st,premultipliedAlpha:x.premultipliedAlpha,doubleSided:x.side===Rn,flipSided:x.side===Jt,useDepthPacking:x.depthPacking>=0,depthPacking:x.depthPacking||0,index0AttributeName:x.index0AttributeName,extensionClipCullDistance:me&&x.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(me&&x.extensions.multiDraw===!0||Ce)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:x.customProgramCacheKey()};return Me.vertexUv1s=l.has(1),Me.vertexUv2s=l.has(2),Me.vertexUv3s=l.has(3),l.clear(),Me}function m(x){let E=[];if(x.shaderID?E.push(x.shaderID):(E.push(x.customVertexShaderID),E.push(x.customFragmentShaderID)),x.defines!==void 0)for(let R in x.defines)E.push(R),E.push(x.defines[R]);return x.isRawShaderMaterial===!1&&(f(E,x),g(E,x),E.push(t.outputColorSpace)),E.push(x.customProgramCacheKey),E.join()}function f(x,E){x.push(E.precision),x.push(E.outputColorSpace),x.push(E.envMapMode),x.push(E.envMapCubeUVHeight),x.push(E.mapUv),x.push(E.alphaMapUv),x.push(E.lightMapUv),x.push(E.aoMapUv),x.push(E.bumpMapUv),x.push(E.normalMapUv),x.push(E.displacementMapUv),x.push(E.emissiveMapUv),x.push(E.metalnessMapUv),x.push(E.roughnessMapUv),x.push(E.anisotropyMapUv),x.push(E.clearcoatMapUv),x.push(E.clearcoatNormalMapUv),x.push(E.clearcoatRoughnessMapUv),x.push(E.iridescenceMapUv),x.push(E.iridescenceThicknessMapUv),x.push(E.sheenColorMapUv),x.push(E.sheenRoughnessMapUv),x.push(E.specularMapUv),x.push(E.specularColorMapUv),x.push(E.specularIntensityMapUv),x.push(E.transmissionMapUv),x.push(E.thicknessMapUv),x.push(E.combine),x.push(E.fogExp2),x.push(E.sizeAttenuation),x.push(E.morphTargetsCount),x.push(E.morphAttributeCount),x.push(E.numDirLights),x.push(E.numPointLights),x.push(E.numSpotLights),x.push(E.numSpotLightMaps),x.push(E.numHemiLights),x.push(E.numRectAreaLights),x.push(E.numDirLightShadows),x.push(E.numPointLightShadows),x.push(E.numSpotLightShadows),x.push(E.numSpotLightShadowsWithMaps),x.push(E.numLightProbes),x.push(E.shadowMapType),x.push(E.toneMapping),x.push(E.numClippingPlanes),x.push(E.numClipIntersection),x.push(E.depthPacking)}function g(x,E){a.disableAll(),E.instancing&&a.enable(0),E.instancingColor&&a.enable(1),E.instancingMorph&&a.enable(2),E.matcap&&a.enable(3),E.envMap&&a.enable(4),E.normalMapObjectSpace&&a.enable(5),E.normalMapTangentSpace&&a.enable(6),E.clearcoat&&a.enable(7),E.iridescence&&a.enable(8),E.alphaTest&&a.enable(9),E.vertexColors&&a.enable(10),E.vertexAlphas&&a.enable(11),E.vertexUv1s&&a.enable(12),E.vertexUv2s&&a.enable(13),E.vertexUv3s&&a.enable(14),E.vertexTangents&&a.enable(15),E.anisotropy&&a.enable(16),E.alphaHash&&a.enable(17),E.batching&&a.enable(18),E.dispersion&&a.enable(19),E.batchingColor&&a.enable(20),E.gradientMap&&a.enable(21),E.packedNormalMap&&a.enable(22),E.vertexNormals&&a.enable(23),x.push(a.mask),a.disableAll(),E.fog&&a.enable(0),E.useFog&&a.enable(1),E.flatShading&&a.enable(2),E.logarithmicDepthBuffer&&a.enable(3),E.reversedDepthBuffer&&a.enable(4),E.skinning&&a.enable(5),E.morphTargets&&a.enable(6),E.morphNormals&&a.enable(7),E.morphColors&&a.enable(8),E.premultipliedAlpha&&a.enable(9),E.shadowMapEnabled&&a.enable(10),E.doubleSided&&a.enable(11),E.flipSided&&a.enable(12),E.useDepthPacking&&a.enable(13),E.dithering&&a.enable(14),E.transmission&&a.enable(15),E.sheen&&a.enable(16),E.opaque&&a.enable(17),E.pointsUvs&&a.enable(18),E.decodeVideoTexture&&a.enable(19),E.decodeVideoTextureEmissive&&a.enable(20),E.alphaToCoverage&&a.enable(21),E.numLightProbeGrids>0&&a.enable(22),E.hasPositionAttribute&&a.enable(23),x.push(a.mask)}function S(x){let E=d[x.type],R;if(E){let D=ji[E];R=sA.clone(D.uniforms)}else R=x.uniforms;return R}function _(x,E){let R=h.get(E);return R!==void 0?++R.usedTimes:(R=new X2(t,E,x,s),c.push(R),h.set(E,R)),R}function T(x){if(--x.usedTimes===0){let E=c.indexOf(x);c[E]=c[c.length-1],c.pop(),h.delete(x.cacheKey),x.destroy()}}function b(x){o.remove(x)}function w(){o.dispose()}return{getParameters:M,getProgramCacheKey:m,getUniforms:S,acquireProgram:_,releaseProgram:T,releaseShaderCache:b,programs:c,dispose:w}}function Z2(){let t=new WeakMap;function e(a){return t.has(a)}function n(a){let o=t.get(a);return o===void 0&&(o={},t.set(a,o)),o}function i(a){t.delete(a)}function s(a,o,l){t.get(a)[o]=l}function r(){t=new WeakMap}return{has:e,get:n,remove:i,update:s,dispose:r}}function K2(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.material.id!==e.material.id?t.material.id-e.material.id:t.materialVariant!==e.materialVariant?t.materialVariant-e.materialVariant:t.z!==e.z?t.z-e.z:t.id-e.id}function EA(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function TA(){let t=[],e=0,n=[],i=[],s=[];function r(){e=0,n.length=0,i.length=0,s.length=0}function a(u){let d=0;return u.isInstancedMesh&&(d+=2),u.isSkinnedMesh&&(d+=1),d}function o(u,d,v,M,m,f){let g=t[e];return g===void 0?(g={id:u.id,object:u,geometry:d,material:v,materialVariant:a(u),groupOrder:M,renderOrder:u.renderOrder,z:m,group:f},t[e]=g):(g.id=u.id,g.object=u,g.geometry=d,g.material=v,g.materialVariant=a(u),g.groupOrder=M,g.renderOrder=u.renderOrder,g.z=m,g.group=f),e++,g}function l(u,d,v,M,m,f){let g=o(u,d,v,M,m,f);v.transmission>0?i.push(g):v.transparent===!0?s.push(g):n.push(g)}function c(u,d,v,M,m,f){let g=o(u,d,v,M,m,f);v.transmission>0?i.unshift(g):v.transparent===!0?s.unshift(g):n.unshift(g)}function h(u,d,v){n.length>1&&n.sort(u||K2),i.length>1&&i.sort(d||EA),s.length>1&&s.sort(d||EA),v&&(n.reverse(),i.reverse(),s.reverse())}function p(){for(let u=e,d=t.length;u<d;u++){let v=t[u];if(v.id===null)break;v.id=null,v.object=null,v.geometry=null,v.material=null,v.group=null}}return{opaque:n,transmissive:i,transparent:s,init:r,push:l,unshift:c,finish:p,sort:h}}function J2(){let t=new WeakMap;function e(i,s){let r=t.get(i),a;return r===void 0?(a=new TA,t.set(i,[a])):s>=r.length?(a=new TA,r.push(a)):a=r[s],a}function n(){t=new WeakMap}return{get:e,dispose:n}}function j2(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={direction:new z,color:new ke};break;case"SpotLight":n={position:new z,direction:new z,color:new ke,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new z,color:new ke,distance:0,decay:0};break;case"HemisphereLight":n={direction:new z,skyColor:new ke,groundColor:new ke};break;case"RectAreaLight":n={color:new ke,position:new z,halfWidth:new z,halfHeight:new z};break}return t[e.id]=n,n}}}function $2(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ie};break;case"SpotLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ie};break;case"PointLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ie,shadowCameraNear:1,shadowCameraFar:1e3};break}return t[e.id]=n,n}}}var eR=0;function tR(t,e){return(e.castShadow?2:0)-(t.castShadow?2:0)+(e.map?1:0)-(t.map?1:0)}function nR(t){let e=new j2,n=$2(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)i.probe.push(new z);let s=new z,r=new Ht,a=new Ht;function o(c){let h=0,p=0,u=0;for(let E=0;E<9;E++)i.probe[E].set(0,0,0);let d=0,v=0,M=0,m=0,f=0,g=0,S=0,_=0,T=0,b=0,w=0;c.sort(tR);for(let E=0,R=c.length;E<R;E++){let D=c[E],L=D.color,q=D.intensity,Y=D.distance,N=null;if(D.shadow&&D.shadow.map&&(D.shadow.map.texture.format===Sr?N=D.shadow.map.texture:N=D.shadow.map.depthTexture||D.shadow.map.texture),D.isAmbientLight)h+=L.r*q,p+=L.g*q,u+=L.b*q;else if(D.isLightProbe){for(let k=0;k<9;k++)i.probe[k].addScaledVector(D.sh.coefficients[k],q);w++}else if(D.isDirectionalLight){let k=e.get(D);if(k.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){let V=D.shadow,j=n.get(D);j.shadowIntensity=V.intensity,j.shadowBias=V.bias,j.shadowNormalBias=V.normalBias,j.shadowRadius=V.radius,j.shadowMapSize=V.mapSize,i.directionalShadow[d]=j,i.directionalShadowMap[d]=N,i.directionalShadowMatrix[d]=D.shadow.matrix,g++}i.directional[d]=k,d++}else if(D.isSpotLight){let k=e.get(D);k.position.setFromMatrixPosition(D.matrixWorld),k.color.copy(L).multiplyScalar(q),k.distance=Y,k.coneCos=Math.cos(D.angle),k.penumbraCos=Math.cos(D.angle*(1-D.penumbra)),k.decay=D.decay,i.spot[M]=k;let V=D.shadow;if(D.map&&(i.spotLightMap[T]=D.map,T++,V.updateMatrices(D),D.castShadow&&b++),i.spotLightMatrix[M]=V.matrix,D.castShadow){let j=n.get(D);j.shadowIntensity=V.intensity,j.shadowBias=V.bias,j.shadowNormalBias=V.normalBias,j.shadowRadius=V.radius,j.shadowMapSize=V.mapSize,i.spotShadow[M]=j,i.spotShadowMap[M]=N,_++}M++}else if(D.isRectAreaLight){let k=e.get(D);k.color.copy(L).multiplyScalar(q),k.halfWidth.set(D.width*.5,0,0),k.halfHeight.set(0,D.height*.5,0),i.rectArea[m]=k,m++}else if(D.isPointLight){let k=e.get(D);if(k.color.copy(D.color).multiplyScalar(D.intensity),k.distance=D.distance,k.decay=D.decay,D.castShadow){let V=D.shadow,j=n.get(D);j.shadowIntensity=V.intensity,j.shadowBias=V.bias,j.shadowNormalBias=V.normalBias,j.shadowRadius=V.radius,j.shadowMapSize=V.mapSize,j.shadowCameraNear=V.camera.near,j.shadowCameraFar=V.camera.far,i.pointShadow[v]=j,i.pointShadowMap[v]=N,i.pointShadowMatrix[v]=D.shadow.matrix,S++}i.point[v]=k,v++}else if(D.isHemisphereLight){let k=e.get(D);k.skyColor.copy(D.color).multiplyScalar(q),k.groundColor.copy(D.groundColor).multiplyScalar(q),i.hemi[f]=k,f++}}m>0&&(t.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=fe.LTC_FLOAT_1,i.rectAreaLTC2=fe.LTC_FLOAT_2):(i.rectAreaLTC1=fe.LTC_HALF_1,i.rectAreaLTC2=fe.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=p,i.ambient[2]=u;let x=i.hash;(x.directionalLength!==d||x.pointLength!==v||x.spotLength!==M||x.rectAreaLength!==m||x.hemiLength!==f||x.numDirectionalShadows!==g||x.numPointShadows!==S||x.numSpotShadows!==_||x.numSpotMaps!==T||x.numLightProbes!==w)&&(i.directional.length=d,i.spot.length=M,i.rectArea.length=m,i.point.length=v,i.hemi.length=f,i.directionalShadow.length=g,i.directionalShadowMap.length=g,i.pointShadow.length=S,i.pointShadowMap.length=S,i.spotShadow.length=_,i.spotShadowMap.length=_,i.directionalShadowMatrix.length=g,i.pointShadowMatrix.length=S,i.spotLightMatrix.length=_+T-b,i.spotLightMap.length=T,i.numSpotLightShadowsWithMaps=b,i.numLightProbes=w,x.directionalLength=d,x.pointLength=v,x.spotLength=M,x.rectAreaLength=m,x.hemiLength=f,x.numDirectionalShadows=g,x.numPointShadows=S,x.numSpotShadows=_,x.numSpotMaps=T,x.numLightProbes=w,i.version=eR++)}function l(c,h){let p=0,u=0,d=0,v=0,M=0,m=h.matrixWorldInverse;for(let f=0,g=c.length;f<g;f++){let S=c[f];if(S.isDirectionalLight){let _=i.directional[p];_.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),_.direction.sub(s),_.direction.transformDirection(m),p++}else if(S.isSpotLight){let _=i.spot[d];_.position.setFromMatrixPosition(S.matrixWorld),_.position.applyMatrix4(m),_.direction.setFromMatrixPosition(S.matrixWorld),s.setFromMatrixPosition(S.target.matrixWorld),_.direction.sub(s),_.direction.transformDirection(m),d++}else if(S.isRectAreaLight){let _=i.rectArea[v];_.position.setFromMatrixPosition(S.matrixWorld),_.position.applyMatrix4(m),a.identity(),r.copy(S.matrixWorld),r.premultiply(m),a.extractRotation(r),_.halfWidth.set(S.width*.5,0,0),_.halfHeight.set(0,S.height*.5,0),_.halfWidth.applyMatrix4(a),_.halfHeight.applyMatrix4(a),v++}else if(S.isPointLight){let _=i.point[u];_.position.setFromMatrixPosition(S.matrixWorld),_.position.applyMatrix4(m),u++}else if(S.isHemisphereLight){let _=i.hemi[M];_.direction.setFromMatrixPosition(S.matrixWorld),_.direction.transformDirection(m),M++}}}return{setup:o,setupView:l,state:i}}function bA(t){let e=new nR(t),n=[],i=[],s=[];function r(u){p.camera=u,n.length=0,i.length=0,s.length=0}function a(u){n.push(u)}function o(u){i.push(u)}function l(u){s.push(u)}function c(){e.setup(n)}function h(u){e.setupView(n,u)}let p={lightsArray:n,shadowsArray:i,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:p,setupLights:c,setupLightsView:h,pushLight:a,pushShadow:o,pushLightProbeGrid:l}}function iR(t){let e=new WeakMap;function n(s,r=0){let a=e.get(s),o;return a===void 0?(o=new bA(t),e.set(s,[o])):r>=a.length?(o=new bA(t),a.push(o)):o=a[r],o}function i(){e=new WeakMap}return{get:n,dispose:i}}var sR=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,rR=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,aR=[new z(1,0,0),new z(-1,0,0),new z(0,1,0),new z(0,-1,0),new z(0,0,1),new z(0,0,-1)],oR=[new z(0,-1,0),new z(0,-1,0),new z(0,0,1),new z(0,0,-1),new z(0,-1,0),new z(0,-1,0)],wA=new Ht,ac=new z,Hg=new z;function lR(t,e,n){let i=new Vl,s=new Ie,r=new Ie,a=new Dt,o=new Cf,l=new Rf,c={},h=n.maxTextureSize,p={[wi]:Jt,[Jt]:wi,[Rn]:Rn},u=new Vt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ie},radius:{value:4}},vertexShader:sR,fragmentShader:rR}),d=u.clone();d.defines.HORIZONTAL_PASS=1;let v=new xi;v.setAttribute("position",new An(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let M=new Mn(v,u),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ql;let f=this.type;this.render=function(b,w,x){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||b.length===0)return;this.type===ES&&(Re("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=Ql);let E=t.getRenderTarget(),R=t.getActiveCubeFace(),D=t.getActiveMipmapLevel(),L=t.state;L.setBlending(Gn),L.buffers.depth.getReversed()===!0?L.buffers.color.setClear(0,0,0,0):L.buffers.color.setClear(1,1,1,1),L.buffers.depth.setTest(!0),L.setScissorTest(!1);let q=f!==this.type;q&&w.traverse(function(Y){Y.material&&(Array.isArray(Y.material)?Y.material.forEach(N=>N.needsUpdate=!0):Y.material.needsUpdate=!0)});for(let Y=0,N=b.length;Y<N;Y++){let k=b[Y],V=k.shadow;if(V===void 0){Re("WebGLShadowMap:",k,"has no shadow.");continue}if(V.autoUpdate===!1&&V.needsUpdate===!1)continue;s.copy(V.mapSize);let j=V.getFrameExtents();s.multiply(j),r.copy(V.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/j.x),s.x=r.x*j.x,V.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/j.y),s.y=r.y*j.y,V.mapSize.y=r.y));let ee=t.state.buffers.depth.getReversed();if(V.camera._reversedDepth=ee,V.map===null||q===!0){if(V.map!==null&&(V.map.depthTexture!==null&&(V.map.depthTexture.dispose(),V.map.depthTexture=null),V.map.dispose()),this.type===So){if(k.isPointLight){Re("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}V.map=new Nt(s.x,s.y,{format:Sr,type:Qi,minFilter:xt,magFilter:xt,generateMipmaps:!1}),V.map.texture.name=k.name+".shadowMap",V.map.depthTexture=new Ri(s.x,s.y,ti),V.map.depthTexture.name=k.name+".shadowMapDepth",V.map.depthTexture.format=Yi,V.map.depthTexture.compareFunction=null,V.map.depthTexture.minFilter=un,V.map.depthTexture.magFilter=un}else k.isPointLight?(V.map=new Ih(s.x),V.map.depthTexture=new bf(s.x,Ui)):(V.map=new Nt(s.x,s.y),V.map.depthTexture=new Ri(s.x,s.y,Ui)),V.map.depthTexture.name=k.name+".shadowMap",V.map.depthTexture.format=Yi,this.type===Ql?(V.map.depthTexture.compareFunction=ee?Rh:Ch,V.map.depthTexture.minFilter=xt,V.map.depthTexture.magFilter=xt):(V.map.depthTexture.compareFunction=null,V.map.depthTexture.minFilter=un,V.map.depthTexture.magFilter=un);V.camera.updateProjectionMatrix()}let se=V.map.isWebGLCubeRenderTarget?6:1;for(let he=0;he<se;he++){if(V.map.isWebGLCubeRenderTarget)t.setRenderTarget(V.map,he),t.clear();else{he===0&&(t.setRenderTarget(V.map),t.clear());let ve=V.getViewport(he);a.set(r.x*ve.x,r.y*ve.y,r.x*ve.z,r.y*ve.w),L.viewport(a)}if(k.isPointLight){let ve=V.camera,Ke=V.matrix,yt=k.distance||ve.far;yt!==ve.far&&(ve.far=yt,ve.updateProjectionMatrix()),ac.setFromMatrixPosition(k.matrixWorld),ve.position.copy(ac),Hg.copy(ve.position),Hg.add(aR[he]),ve.up.copy(oR[he]),ve.lookAt(Hg),ve.updateMatrixWorld(),Ke.makeTranslation(-ac.x,-ac.y,-ac.z),wA.multiplyMatrices(ve.projectionMatrix,ve.matrixWorldInverse),V._frustum.setFromProjectionMatrix(wA,ve.coordinateSystem,ve.reversedDepth)}else V.updateMatrices(k);i=V.getFrustum(),_(w,x,V.camera,k,this.type)}V.isPointLightShadow!==!0&&this.type===So&&g(V,x),V.needsUpdate=!1}f=this.type,m.needsUpdate=!1,t.setRenderTarget(E,R,D)};function g(b,w){let x=e.update(M);u.defines.VSM_SAMPLES!==b.blurSamples&&(u.defines.VSM_SAMPLES=b.blurSamples,d.defines.VSM_SAMPLES=b.blurSamples,u.needsUpdate=!0,d.needsUpdate=!0),b.mapPass===null&&(b.mapPass=new Nt(s.x,s.y,{format:Sr,type:Qi})),u.uniforms.shadow_pass.value=b.map.depthTexture,u.uniforms.resolution.value=b.mapSize,u.uniforms.radius.value=b.radius,t.setRenderTarget(b.mapPass),t.clear(),t.renderBufferDirect(w,null,x,u,M,null),d.uniforms.shadow_pass.value=b.mapPass.texture,d.uniforms.resolution.value=b.mapSize,d.uniforms.radius.value=b.radius,t.setRenderTarget(b.map),t.clear(),t.renderBufferDirect(w,null,x,d,M,null)}function S(b,w,x,E){let R=null,D=x.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(D!==void 0)R=D;else if(R=x.isPointLight===!0?l:o,t.localClippingEnabled&&w.clipShadows===!0&&Array.isArray(w.clippingPlanes)&&w.clippingPlanes.length!==0||w.displacementMap&&w.displacementScale!==0||w.alphaMap&&w.alphaTest>0||w.map&&w.alphaTest>0||w.alphaToCoverage===!0){let L=R.uuid,q=w.uuid,Y=c[L];Y===void 0&&(Y={},c[L]=Y);let N=Y[q];N===void 0&&(N=R.clone(),Y[q]=N,w.addEventListener("dispose",T)),R=N}if(R.visible=w.visible,R.wireframe=w.wireframe,E===So?R.side=w.shadowSide!==null?w.shadowSide:w.side:R.side=w.shadowSide!==null?w.shadowSide:p[w.side],R.alphaMap=w.alphaMap,R.alphaTest=w.alphaToCoverage===!0?.5:w.alphaTest,R.map=w.map,R.clipShadows=w.clipShadows,R.clippingPlanes=w.clippingPlanes,R.clipIntersection=w.clipIntersection,R.displacementMap=w.displacementMap,R.displacementScale=w.displacementScale,R.displacementBias=w.displacementBias,R.wireframeLinewidth=w.wireframeLinewidth,R.linewidth=w.linewidth,x.isPointLight===!0&&R.isMeshDistanceMaterial===!0){let L=t.properties.get(R);L.light=x}return R}function _(b,w,x,E,R){if(b.visible===!1)return;if(b.layers.test(w.layers)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&R===So)&&(!b.frustumCulled||i.intersectsObject(b))){b.modelViewMatrix.multiplyMatrices(x.matrixWorldInverse,b.matrixWorld);let q=e.update(b),Y=b.material;if(Array.isArray(Y)){let N=q.groups;for(let k=0,V=N.length;k<V;k++){let j=N[k],ee=Y[j.materialIndex];if(ee&&ee.visible){let se=S(b,ee,E,R);b.onBeforeShadow(t,b,w,x,q,se,j),t.renderBufferDirect(x,null,q,se,b,j),b.onAfterShadow(t,b,w,x,q,se,j)}}}else if(Y.visible){let N=S(b,Y,E,R);b.onBeforeShadow(t,b,w,x,q,N,null),t.renderBufferDirect(x,null,q,N,b,null),b.onAfterShadow(t,b,w,x,q,N,null)}}let L=b.children;for(let q=0,Y=L.length;q<Y;q++)_(L[q],w,x,E,R)}function T(b){b.target.removeEventListener("dispose",T);for(let x in c){let E=c[x],R=b.target.uuid;R in E&&(E[R].dispose(),delete E[R])}}}function cR(t,e){function n(){let U=!1,re=new Dt,Q=null,ce=new Dt(0,0,0,0);return{setMask:function(me){Q!==me&&!U&&(t.colorMask(me,me,me,me),Q=me)},setLocked:function(me){U=me},setClear:function(me,$,Me,ye,Bt){Bt===!0&&(me*=ye,$*=ye,Me*=ye),re.set(me,$,Me,ye),ce.equals(re)===!1&&(t.clearColor(me,$,Me,ye),ce.copy(re))},reset:function(){U=!1,Q=null,ce.set(-1,0,0,0)}}}function i(){let U=!1,re=!1,Q=null,ce=null,me=null;return{setReversed:function($){if(re!==$){let Me=e.get("EXT_clip_control");$?Me.clipControlEXT(Me.LOWER_LEFT_EXT,Me.ZERO_TO_ONE_EXT):Me.clipControlEXT(Me.LOWER_LEFT_EXT,Me.NEGATIVE_ONE_TO_ONE_EXT),re=$;let ye=me;me=null,this.setClear(ye)}},getReversed:function(){return re},setTest:function($){$?te(t.DEPTH_TEST):Se(t.DEPTH_TEST)},setMask:function($){Q!==$&&!U&&(t.depthMask($),Q=$)},setFunc:function($){if(re&&($=nA[$]),ce!==$){switch($){case ff:t.depthFunc(t.NEVER);break;case mo:t.depthFunc(t.ALWAYS);break;case hf:t.depthFunc(t.LESS);break;case ea:t.depthFunc(t.LEQUAL);break;case df:t.depthFunc(t.EQUAL);break;case pf:t.depthFunc(t.GEQUAL);break;case mf:t.depthFunc(t.GREATER);break;case gf:t.depthFunc(t.NOTEQUAL);break;default:t.depthFunc(t.LEQUAL)}ce=$}},setLocked:function($){U=$},setClear:function($){me!==$&&(me=$,re&&($=1-$),t.clearDepth($))},reset:function(){U=!1,Q=null,ce=null,me=null,re=!1}}}function s(){let U=!1,re=null,Q=null,ce=null,me=null,$=null,Me=null,ye=null,Bt=null;return{setTest:function(mt){U||(mt?te(t.STENCIL_TEST):Se(t.STENCIL_TEST))},setMask:function(mt){re!==mt&&!U&&(t.stencilMask(mt),re=mt)},setFunc:function(mt,Bi,Ii){(Q!==mt||ce!==Bi||me!==Ii)&&(t.stencilFunc(mt,Bi,Ii),Q=mt,ce=Bi,me=Ii)},setOp:function(mt,Bi,Ii){($!==mt||Me!==Bi||ye!==Ii)&&(t.stencilOp(mt,Bi,Ii),$=mt,Me=Bi,ye=Ii)},setLocked:function(mt){U=mt},setClear:function(mt){Bt!==mt&&(t.clearStencil(mt),Bt=mt)},reset:function(){U=!1,re=null,Q=null,ce=null,me=null,$=null,Me=null,ye=null,Bt=null}}}let r=new n,a=new i,o=new s,l=new WeakMap,c=new WeakMap,h={},p={},u={},d=new WeakMap,v=[],M=null,m=!1,f=null,g=null,S=null,_=null,T=null,b=null,w=null,x=new ke(0,0,0),E=0,R=!1,D=null,L=null,q=null,Y=null,N=null,k=t.getParameter(t.MAX_COMBINED_TEXTURE_IMAGE_UNITS),V=!1,j=0,ee=t.getParameter(t.VERSION);ee.indexOf("WebGL")!==-1?(j=parseFloat(/^WebGL (\d)/.exec(ee)[1]),V=j>=1):ee.indexOf("OpenGL ES")!==-1&&(j=parseFloat(/^OpenGL ES (\d)/.exec(ee)[1]),V=j>=2);let se=null,he={},ve=t.getParameter(t.SCISSOR_BOX),Ke=t.getParameter(t.VIEWPORT),yt=new Dt().fromArray(ve),Je=new Dt().fromArray(Ke);function Z(U,re,Q,ce){let me=new Uint8Array(4),$=t.createTexture();t.bindTexture(U,$),t.texParameteri(U,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(U,t.TEXTURE_MAG_FILTER,t.NEAREST);for(let Me=0;Me<Q;Me++)U===t.TEXTURE_3D||U===t.TEXTURE_2D_ARRAY?t.texImage3D(re,0,t.RGBA,1,1,ce,0,t.RGBA,t.UNSIGNED_BYTE,me):t.texImage2D(re+Me,0,t.RGBA,1,1,0,t.RGBA,t.UNSIGNED_BYTE,me);return $}let ie={};ie[t.TEXTURE_2D]=Z(t.TEXTURE_2D,t.TEXTURE_2D,1),ie[t.TEXTURE_CUBE_MAP]=Z(t.TEXTURE_CUBE_MAP,t.TEXTURE_CUBE_MAP_POSITIVE_X,6),ie[t.TEXTURE_2D_ARRAY]=Z(t.TEXTURE_2D_ARRAY,t.TEXTURE_2D_ARRAY,1,1),ie[t.TEXTURE_3D]=Z(t.TEXTURE_3D,t.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),te(t.DEPTH_TEST),a.setFunc(ea),kt(!1),$t(rg),te(t.CULL_FACE),tt(Gn);function te(U){h[U]!==!0&&(t.enable(U),h[U]=!0)}function Se(U){h[U]!==!1&&(t.disable(U),h[U]=!1)}function Ae(U,re){return u[U]!==re?(t.bindFramebuffer(U,re),u[U]=re,U===t.DRAW_FRAMEBUFFER&&(u[t.FRAMEBUFFER]=re),U===t.FRAMEBUFFER&&(u[t.DRAW_FRAMEBUFFER]=re),!0):!1}function Ce(U,re){let Q=v,ce=!1;if(U){Q=d.get(re),Q===void 0&&(Q=[],d.set(re,Q));let me=U.textures;if(Q.length!==me.length||Q[0]!==t.COLOR_ATTACHMENT0){for(let $=0,Me=me.length;$<Me;$++)Q[$]=t.COLOR_ATTACHMENT0+$;Q.length=me.length,ce=!0}}else Q[0]!==t.BACK&&(Q[0]=t.BACK,ce=!0);ce&&t.drawBuffers(Q)}function Rt(U){return M!==U?(t.useProgram(U),M=U,!0):!1}let We={[fr]:t.FUNC_ADD,[bS]:t.FUNC_SUBTRACT,[wS]:t.FUNC_REVERSE_SUBTRACT};We[CS]=t.MIN,We[RS]=t.MAX;let ft={[DS]:t.ZERO,[US]:t.ONE,[BS]:t.SRC_COLOR,[cf]:t.SRC_ALPHA,[FS]:t.SRC_ALPHA_SATURATE,[NS]:t.DST_COLOR,[PS]:t.DST_ALPHA,[IS]:t.ONE_MINUS_SRC_COLOR,[uf]:t.ONE_MINUS_SRC_ALPHA,[OS]:t.ONE_MINUS_DST_COLOR,[LS]:t.ONE_MINUS_DST_ALPHA,[zS]:t.CONSTANT_COLOR,[GS]:t.ONE_MINUS_CONSTANT_COLOR,[HS]:t.CONSTANT_ALPHA,[VS]:t.ONE_MINUS_CONSTANT_ALPHA};function tt(U,re,Q,ce,me,$,Me,ye,Bt,mt){if(U===Gn){m===!0&&(Se(t.BLEND),m=!1);return}if(m===!1&&(te(t.BLEND),m=!0),U!==TS){if(U!==f||mt!==R){if((g!==fr||T!==fr)&&(t.blendEquation(t.FUNC_ADD),g=fr,T=fr),mt)switch(U){case $r:t.blendFuncSeparate(t.ONE,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case ag:t.blendFunc(t.ONE,t.ONE);break;case og:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case lg:t.blendFuncSeparate(t.DST_COLOR,t.ONE_MINUS_SRC_ALPHA,t.ZERO,t.ONE);break;default:Ue("WebGLState: Invalid blending: ",U);break}else switch(U){case $r:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case ag:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE,t.ONE,t.ONE);break;case og:Ue("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case lg:Ue("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Ue("WebGLState: Invalid blending: ",U);break}S=null,_=null,b=null,w=null,x.set(0,0,0),E=0,f=U,R=mt}return}me=me||re,$=$||Q,Me=Me||ce,(re!==g||me!==T)&&(t.blendEquationSeparate(We[re],We[me]),g=re,T=me),(Q!==S||ce!==_||$!==b||Me!==w)&&(t.blendFuncSeparate(ft[Q],ft[ce],ft[$],ft[Me]),S=Q,_=ce,b=$,w=Me),(ye.equals(x)===!1||Bt!==E)&&(t.blendColor(ye.r,ye.g,ye.b,Bt),x.copy(ye),E=Bt),f=U,R=!1}function je(U,re){U.side===Rn?Se(t.CULL_FACE):te(t.CULL_FACE);let Q=U.side===Jt;re&&(Q=!Q),kt(Q),U.blending===$r&&U.transparent===!1?tt(Gn):tt(U.blending,U.blendEquation,U.blendSrc,U.blendDst,U.blendEquationAlpha,U.blendSrcAlpha,U.blendDstAlpha,U.blendColor,U.blendAlpha,U.premultipliedAlpha),a.setFunc(U.depthFunc),a.setTest(U.depthTest),a.setMask(U.depthWrite),r.setMask(U.colorWrite);let ce=U.stencilWrite;o.setTest(ce),ce&&(o.setMask(U.stencilWriteMask),o.setFunc(U.stencilFunc,U.stencilRef,U.stencilFuncMask),o.setOp(U.stencilFail,U.stencilZFail,U.stencilZPass)),fn(U.polygonOffset,U.polygonOffsetFactor,U.polygonOffsetUnits),U.alphaToCoverage===!0?te(t.SAMPLE_ALPHA_TO_COVERAGE):Se(t.SAMPLE_ALPHA_TO_COVERAGE)}function kt(U){D!==U&&(U?t.frontFace(t.CW):t.frontFace(t.CCW),D=U)}function $t(U){U!==AS?(te(t.CULL_FACE),U!==L&&(U===rg?t.cullFace(t.BACK):U===MS?t.cullFace(t.FRONT):t.cullFace(t.FRONT_AND_BACK))):Se(t.CULL_FACE),L=U}function rn(U){U!==q&&(V&&t.lineWidth(U),q=U)}function fn(U,re,Q){U?(te(t.POLYGON_OFFSET_FILL),(Y!==re||N!==Q)&&(Y=re,N=Q,a.getReversed()&&(re=-re),t.polygonOffset(re,Q))):Se(t.POLYGON_OFFSET_FILL)}function Ut(U){U?te(t.SCISSOR_TEST):Se(t.SCISSOR_TEST)}function Wt(U){U===void 0&&(U=t.TEXTURE0+k-1),se!==U&&(t.activeTexture(U),se=U)}function B(U,re,Q){Q===void 0&&(se===null?Q=t.TEXTURE0+k-1:Q=se);let ce=he[Q];ce===void 0&&(ce={type:void 0,texture:void 0},he[Q]=ce),(ce.type!==U||ce.texture!==re)&&(se!==Q&&(t.activeTexture(Q),se=Q),t.bindTexture(U,re||ie[U]),ce.type=U,ce.texture=re)}function Dn(){let U=he[se];U!==void 0&&U.type!==void 0&&(t.bindTexture(U.type,null),U.type=void 0,U.texture=void 0)}function rt(){try{t.compressedTexImage2D(...arguments)}catch(U){Ue("WebGLState:",U)}}function C(){try{t.compressedTexImage3D(...arguments)}catch(U){Ue("WebGLState:",U)}}function y(){try{t.texSubImage2D(...arguments)}catch(U){Ue("WebGLState:",U)}}function P(){try{t.texSubImage3D(...arguments)}catch(U){Ue("WebGLState:",U)}}function G(){try{t.compressedTexSubImage2D(...arguments)}catch(U){Ue("WebGLState:",U)}}function W(){try{t.compressedTexSubImage3D(...arguments)}catch(U){Ue("WebGLState:",U)}}function ne(){try{t.texStorage2D(...arguments)}catch(U){Ue("WebGLState:",U)}}function ae(){try{t.texStorage3D(...arguments)}catch(U){Ue("WebGLState:",U)}}function X(){try{t.texImage2D(...arguments)}catch(U){Ue("WebGLState:",U)}}function K(){try{t.texImage3D(...arguments)}catch(U){Ue("WebGLState:",U)}}function oe(U){return p[U]!==void 0?p[U]:t.getParameter(U)}function Ee(U,re){p[U]!==re&&(t.pixelStorei(U,re),p[U]=re)}function ue(U){yt.equals(U)===!1&&(t.scissor(U.x,U.y,U.z,U.w),yt.copy(U))}function le(U){Je.equals(U)===!1&&(t.viewport(U.x,U.y,U.z,U.w),Je.copy(U))}function we(U,re){let Q=c.get(re);Q===void 0&&(Q=new WeakMap,c.set(re,Q));let ce=Q.get(U);ce===void 0&&(ce=t.getUniformBlockIndex(re,U.name),Q.set(U,ce))}function De(U,re){let ce=c.get(re).get(U);l.get(re)!==ce&&(t.uniformBlockBinding(re,ce,U.__bindingPointIndex),l.set(re,ce))}function Ne(){t.disable(t.BLEND),t.disable(t.CULL_FACE),t.disable(t.DEPTH_TEST),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SCISSOR_TEST),t.disable(t.STENCIL_TEST),t.disable(t.SAMPLE_ALPHA_TO_COVERAGE),t.blendEquation(t.FUNC_ADD),t.blendFunc(t.ONE,t.ZERO),t.blendFuncSeparate(t.ONE,t.ZERO,t.ONE,t.ZERO),t.blendColor(0,0,0,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(t.LESS),a.setReversed(!1),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(t.ALWAYS,0,4294967295),t.stencilOp(t.KEEP,t.KEEP,t.KEEP),t.clearStencil(0),t.cullFace(t.BACK),t.frontFace(t.CCW),t.polygonOffset(0,0),t.activeTexture(t.TEXTURE0),t.bindFramebuffer(t.FRAMEBUFFER,null),t.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),t.bindFramebuffer(t.READ_FRAMEBUFFER,null),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),t.pixelStorei(t.PACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,!1),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,t.BROWSER_DEFAULT_WEBGL),t.pixelStorei(t.PACK_ROW_LENGTH,0),t.pixelStorei(t.PACK_SKIP_PIXELS,0),t.pixelStorei(t.PACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_ROW_LENGTH,0),t.pixelStorei(t.UNPACK_IMAGE_HEIGHT,0),t.pixelStorei(t.UNPACK_SKIP_PIXELS,0),t.pixelStorei(t.UNPACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_SKIP_IMAGES,0),h={},p={},se=null,he={},u={},d=new WeakMap,v=[],M=null,m=!1,f=null,g=null,S=null,_=null,T=null,b=null,w=null,x=new ke(0,0,0),E=0,R=!1,D=null,L=null,q=null,Y=null,N=null,yt.set(0,0,t.canvas.width,t.canvas.height),Je.set(0,0,t.canvas.width,t.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:te,disable:Se,bindFramebuffer:Ae,drawBuffers:Ce,useProgram:Rt,setBlending:tt,setMaterial:je,setFlipSided:kt,setCullFace:$t,setLineWidth:rn,setPolygonOffset:fn,setScissorTest:Ut,activeTexture:Wt,bindTexture:B,unbindTexture:Dn,compressedTexImage2D:rt,compressedTexImage3D:C,texImage2D:X,texImage3D:K,pixelStorei:Ee,getParameter:oe,updateUBOMapping:we,uniformBlockBinding:De,texStorage2D:ne,texStorage3D:ae,texSubImage2D:y,texSubImage3D:P,compressedTexSubImage2D:G,compressedTexSubImage3D:W,scissor:ue,viewport:le,reset:Ne}}function uR(t,e,n,i,s,r,a){let o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Ie,h=new WeakMap,p=new Set,u,d=new WeakMap,v=!1;try{v=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function M(C,y){return v?new OffscreenCanvas(C,y):Nl("canvas")}function m(C,y,P){let G=1,W=rt(C);if((W.width>P||W.height>P)&&(G=P/Math.max(W.width,W.height)),G<1)if(typeof HTMLImageElement<"u"&&C instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&C instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&C instanceof ImageBitmap||typeof VideoFrame<"u"&&C instanceof VideoFrame){let ne=Math.floor(G*W.width),ae=Math.floor(G*W.height);u===void 0&&(u=M(ne,ae));let X=y?M(ne,ae):u;return X.width=ne,X.height=ae,X.getContext("2d").drawImage(C,0,0,ne,ae),Re("WebGLRenderer: Texture has been resized from ("+W.width+"x"+W.height+") to ("+ne+"x"+ae+")."),X}else return"data"in C&&Re("WebGLRenderer: Image in DataTexture is too big ("+W.width+"x"+W.height+")."),C;return C}function f(C){return C.generateMipmaps}function g(C){t.generateMipmap(C)}function S(C){return C.isWebGLCubeRenderTarget?t.TEXTURE_CUBE_MAP:C.isWebGL3DRenderTarget?t.TEXTURE_3D:C.isWebGLArrayRenderTarget||C.isCompressedArrayTexture?t.TEXTURE_2D_ARRAY:t.TEXTURE_2D}function _(C,y,P,G,W,ne=!1){if(C!==null){if(t[C]!==void 0)return t[C];Re("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+C+"'")}let ae;G&&(ae=e.get("EXT_texture_norm16"),ae||Re("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let X=y;if(y===t.RED&&(P===t.FLOAT&&(X=t.R32F),P===t.HALF_FLOAT&&(X=t.R16F),P===t.UNSIGNED_BYTE&&(X=t.R8),P===t.UNSIGNED_SHORT&&ae&&(X=ae.R16_EXT),P===t.SHORT&&ae&&(X=ae.R16_SNORM_EXT)),y===t.RED_INTEGER&&(P===t.UNSIGNED_BYTE&&(X=t.R8UI),P===t.UNSIGNED_SHORT&&(X=t.R16UI),P===t.UNSIGNED_INT&&(X=t.R32UI),P===t.BYTE&&(X=t.R8I),P===t.SHORT&&(X=t.R16I),P===t.INT&&(X=t.R32I)),y===t.RG&&(P===t.FLOAT&&(X=t.RG32F),P===t.HALF_FLOAT&&(X=t.RG16F),P===t.UNSIGNED_BYTE&&(X=t.RG8),P===t.UNSIGNED_SHORT&&ae&&(X=ae.RG16_EXT),P===t.SHORT&&ae&&(X=ae.RG16_SNORM_EXT)),y===t.RG_INTEGER&&(P===t.UNSIGNED_BYTE&&(X=t.RG8UI),P===t.UNSIGNED_SHORT&&(X=t.RG16UI),P===t.UNSIGNED_INT&&(X=t.RG32UI),P===t.BYTE&&(X=t.RG8I),P===t.SHORT&&(X=t.RG16I),P===t.INT&&(X=t.RG32I)),y===t.RGB_INTEGER&&(P===t.UNSIGNED_BYTE&&(X=t.RGB8UI),P===t.UNSIGNED_SHORT&&(X=t.RGB16UI),P===t.UNSIGNED_INT&&(X=t.RGB32UI),P===t.BYTE&&(X=t.RGB8I),P===t.SHORT&&(X=t.RGB16I),P===t.INT&&(X=t.RGB32I)),y===t.RGBA_INTEGER&&(P===t.UNSIGNED_BYTE&&(X=t.RGBA8UI),P===t.UNSIGNED_SHORT&&(X=t.RGBA16UI),P===t.UNSIGNED_INT&&(X=t.RGBA32UI),P===t.BYTE&&(X=t.RGBA8I),P===t.SHORT&&(X=t.RGBA16I),P===t.INT&&(X=t.RGBA32I)),y===t.RGB&&(P===t.UNSIGNED_SHORT&&ae&&(X=ae.RGB16_EXT),P===t.SHORT&&ae&&(X=ae.RGB16_SNORM_EXT),P===t.UNSIGNED_INT_5_9_9_9_REV&&(X=t.RGB9_E5),P===t.UNSIGNED_INT_10F_11F_11F_REV&&(X=t.R11F_G11F_B10F)),y===t.RGBA){let K=ne?Pl:Ye.getTransfer(W);P===t.FLOAT&&(X=t.RGBA32F),P===t.HALF_FLOAT&&(X=t.RGBA16F),P===t.UNSIGNED_BYTE&&(X=K===st?t.SRGB8_ALPHA8:t.RGBA8),P===t.UNSIGNED_SHORT&&ae&&(X=ae.RGBA16_EXT),P===t.SHORT&&ae&&(X=ae.RGBA16_SNORM_EXT),P===t.UNSIGNED_SHORT_4_4_4_4&&(X=t.RGBA4),P===t.UNSIGNED_SHORT_5_5_5_1&&(X=t.RGB5_A1)}return(X===t.R16F||X===t.R32F||X===t.RG16F||X===t.RG32F||X===t.RGBA16F||X===t.RGBA32F)&&e.get("EXT_color_buffer_float"),X}function T(C,y){let P;return C?y===null||y===Ui||y===_r?P=t.DEPTH24_STENCIL8:y===ti?P=t.DEPTH32F_STENCIL8:y===Ao&&(P=t.DEPTH24_STENCIL8,Re("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):y===null||y===Ui||y===_r?P=t.DEPTH_COMPONENT24:y===ti?P=t.DEPTH_COMPONENT32F:y===Ao&&(P=t.DEPTH_COMPONENT16),P}function b(C,y){return f(C)===!0||C.isFramebufferTexture&&C.minFilter!==un&&C.minFilter!==xt?Math.log2(Math.max(y.width,y.height))+1:C.mipmaps!==void 0&&C.mipmaps.length>0?C.mipmaps.length:C.isCompressedTexture&&Array.isArray(C.image)?y.mipmaps.length:1}function w(C){let y=C.target;y.removeEventListener("dispose",w),E(y),y.isVideoTexture&&h.delete(y),y.isHTMLTexture&&p.delete(y)}function x(C){let y=C.target;y.removeEventListener("dispose",x),D(y)}function E(C){let y=i.get(C);if(y.__webglInit===void 0)return;let P=C.source,G=d.get(P);if(G){let W=G[y.__cacheKey];W.usedTimes--,W.usedTimes===0&&R(C),Object.keys(G).length===0&&d.delete(P)}i.remove(C)}function R(C){let y=i.get(C);t.deleteTexture(y.__webglTexture);let P=C.source,G=d.get(P);delete G[y.__cacheKey],a.memory.textures--}function D(C){let y=i.get(C);if(C.depthTexture&&(C.depthTexture.dispose(),i.remove(C.depthTexture)),C.isWebGLCubeRenderTarget)for(let G=0;G<6;G++){if(Array.isArray(y.__webglFramebuffer[G]))for(let W=0;W<y.__webglFramebuffer[G].length;W++)t.deleteFramebuffer(y.__webglFramebuffer[G][W]);else t.deleteFramebuffer(y.__webglFramebuffer[G]);y.__webglDepthbuffer&&t.deleteRenderbuffer(y.__webglDepthbuffer[G])}else{if(Array.isArray(y.__webglFramebuffer))for(let G=0;G<y.__webglFramebuffer.length;G++)t.deleteFramebuffer(y.__webglFramebuffer[G]);else t.deleteFramebuffer(y.__webglFramebuffer);if(y.__webglDepthbuffer&&t.deleteRenderbuffer(y.__webglDepthbuffer),y.__webglMultisampledFramebuffer&&t.deleteFramebuffer(y.__webglMultisampledFramebuffer),y.__webglColorRenderbuffer)for(let G=0;G<y.__webglColorRenderbuffer.length;G++)y.__webglColorRenderbuffer[G]&&t.deleteRenderbuffer(y.__webglColorRenderbuffer[G]);y.__webglDepthRenderbuffer&&t.deleteRenderbuffer(y.__webglDepthRenderbuffer)}let P=C.textures;for(let G=0,W=P.length;G<W;G++){let ne=i.get(P[G]);ne.__webglTexture&&(t.deleteTexture(ne.__webglTexture),a.memory.textures--),i.remove(P[G])}i.remove(C)}let L=0;function q(){L=0}function Y(){return L}function N(C){L=C}function k(){let C=L;return C>=s.maxTextures&&Re("WebGLTextures: Trying to use "+C+" texture units while this GPU supports only "+s.maxTextures),L+=1,C}function V(C){let y=[];return y.push(C.wrapS),y.push(C.wrapT),y.push(C.wrapR||0),y.push(C.magFilter),y.push(C.minFilter),y.push(C.anisotropy),y.push(C.internalFormat),y.push(C.format),y.push(C.type),y.push(C.generateMipmaps),y.push(C.premultiplyAlpha),y.push(C.flipY),y.push(C.unpackAlignment),y.push(C.colorSpace),y.join()}function j(C,y){let P=i.get(C);if(C.isVideoTexture&&B(C),C.isRenderTargetTexture===!1&&C.isExternalTexture!==!0&&C.version>0&&P.__version!==C.version){let G=C.image;if(G===null)Re("WebGLRenderer: Texture marked for update but no image data found.");else if(G.complete===!1)Re("WebGLRenderer: Texture marked for update but image is incomplete");else{Se(P,C,y);return}}else C.isExternalTexture&&(P.__webglTexture=C.sourceTexture?C.sourceTexture:null);n.bindTexture(t.TEXTURE_2D,P.__webglTexture,t.TEXTURE0+y)}function ee(C,y){let P=i.get(C);if(C.isRenderTargetTexture===!1&&C.version>0&&P.__version!==C.version){Se(P,C,y);return}else C.isExternalTexture&&(P.__webglTexture=C.sourceTexture?C.sourceTexture:null);n.bindTexture(t.TEXTURE_2D_ARRAY,P.__webglTexture,t.TEXTURE0+y)}function se(C,y){let P=i.get(C);if(C.isRenderTargetTexture===!1&&C.version>0&&P.__version!==C.version){Se(P,C,y);return}n.bindTexture(t.TEXTURE_3D,P.__webglTexture,t.TEXTURE0+y)}function he(C,y){let P=i.get(C);if(C.isCubeDepthTexture!==!0&&C.version>0&&P.__version!==C.version){Ae(P,C,y);return}n.bindTexture(t.TEXTURE_CUBE_MAP,P.__webglTexture,t.TEXTURE0+y)}let ve={[vf]:t.REPEAT,[Xi]:t.CLAMP_TO_EDGE,[xf]:t.MIRRORED_REPEAT},Ke={[un]:t.NEAREST,[XS]:t.NEAREST_MIPMAP_NEAREST,[Kl]:t.NEAREST_MIPMAP_LINEAR,[xt]:t.LINEAR,[Xf]:t.LINEAR_MIPMAP_NEAREST,[yr]:t.LINEAR_MIPMAP_LINEAR},yt={[qS]:t.NEVER,[jS]:t.ALWAYS,[QS]:t.LESS,[Ch]:t.LEQUAL,[ZS]:t.EQUAL,[Rh]:t.GEQUAL,[KS]:t.GREATER,[JS]:t.NOTEQUAL};function Je(C,y){if(y.type===ti&&e.has("OES_texture_float_linear")===!1&&(y.magFilter===xt||y.magFilter===Xf||y.magFilter===Kl||y.magFilter===yr||y.minFilter===xt||y.minFilter===Xf||y.minFilter===Kl||y.minFilter===yr)&&Re("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),t.texParameteri(C,t.TEXTURE_WRAP_S,ve[y.wrapS]),t.texParameteri(C,t.TEXTURE_WRAP_T,ve[y.wrapT]),(C===t.TEXTURE_3D||C===t.TEXTURE_2D_ARRAY)&&t.texParameteri(C,t.TEXTURE_WRAP_R,ve[y.wrapR]),t.texParameteri(C,t.TEXTURE_MAG_FILTER,Ke[y.magFilter]),t.texParameteri(C,t.TEXTURE_MIN_FILTER,Ke[y.minFilter]),y.compareFunction&&(t.texParameteri(C,t.TEXTURE_COMPARE_MODE,t.COMPARE_REF_TO_TEXTURE),t.texParameteri(C,t.TEXTURE_COMPARE_FUNC,yt[y.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(y.magFilter===un||y.minFilter!==Kl&&y.minFilter!==yr||y.type===ti&&e.has("OES_texture_float_linear")===!1)return;if(y.anisotropy>1||i.get(y).__currentAnisotropy){let P=e.get("EXT_texture_filter_anisotropic");t.texParameterf(C,P.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(y.anisotropy,s.getMaxAnisotropy())),i.get(y).__currentAnisotropy=y.anisotropy}}}function Z(C,y){let P=!1;C.__webglInit===void 0&&(C.__webglInit=!0,y.addEventListener("dispose",w));let G=y.source,W=d.get(G);W===void 0&&(W={},d.set(G,W));let ne=V(y);if(ne!==C.__cacheKey){W[ne]===void 0&&(W[ne]={texture:t.createTexture(),usedTimes:0},a.memory.textures++,P=!0),W[ne].usedTimes++;let ae=W[C.__cacheKey];ae!==void 0&&(W[C.__cacheKey].usedTimes--,ae.usedTimes===0&&R(y)),C.__cacheKey=ne,C.__webglTexture=W[ne].texture}return P}function ie(C,y,P){return Math.floor(Math.floor(C/P)/y)}function te(C,y,P,G){let ne=C.updateRanges;if(ne.length===0)n.texSubImage2D(t.TEXTURE_2D,0,0,0,y.width,y.height,P,G,y.data);else{ne.sort((Ee,ue)=>Ee.start-ue.start);let ae=0;for(let Ee=1;Ee<ne.length;Ee++){let ue=ne[ae],le=ne[Ee],we=ue.start+ue.count,De=ie(le.start,y.width,4),Ne=ie(ue.start,y.width,4);le.start<=we+1&&De===Ne&&ie(le.start+le.count-1,y.width,4)===De?ue.count=Math.max(ue.count,le.start+le.count-ue.start):(++ae,ne[ae]=le)}ne.length=ae+1;let X=n.getParameter(t.UNPACK_ROW_LENGTH),K=n.getParameter(t.UNPACK_SKIP_PIXELS),oe=n.getParameter(t.UNPACK_SKIP_ROWS);n.pixelStorei(t.UNPACK_ROW_LENGTH,y.width);for(let Ee=0,ue=ne.length;Ee<ue;Ee++){let le=ne[Ee],we=Math.floor(le.start/4),De=Math.ceil(le.count/4),Ne=we%y.width,U=Math.floor(we/y.width),re=De,Q=1;n.pixelStorei(t.UNPACK_SKIP_PIXELS,Ne),n.pixelStorei(t.UNPACK_SKIP_ROWS,U),n.texSubImage2D(t.TEXTURE_2D,0,Ne,U,re,Q,P,G,y.data)}C.clearUpdateRanges(),n.pixelStorei(t.UNPACK_ROW_LENGTH,X),n.pixelStorei(t.UNPACK_SKIP_PIXELS,K),n.pixelStorei(t.UNPACK_SKIP_ROWS,oe)}}function Se(C,y,P){let G=t.TEXTURE_2D;(y.isDataArrayTexture||y.isCompressedArrayTexture)&&(G=t.TEXTURE_2D_ARRAY),y.isData3DTexture&&(G=t.TEXTURE_3D);let W=Z(C,y),ne=y.source;n.bindTexture(G,C.__webglTexture,t.TEXTURE0+P);let ae=i.get(ne);if(ne.version!==ae.__version||W===!0){if(n.activeTexture(t.TEXTURE0+P),(typeof ImageBitmap<"u"&&y.image instanceof ImageBitmap)===!1){let Q=Ye.getPrimaries(Ye.workingColorSpace),ce=y.colorSpace===ni?null:Ye.getPrimaries(y.colorSpace),me=y.colorSpace===ni||Q===ce?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,y.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,y.premultiplyAlpha),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,me)}n.pixelStorei(t.UNPACK_ALIGNMENT,y.unpackAlignment);let K=m(y.image,!1,s.maxTextureSize);K=Dn(y,K);let oe=r.convert(y.format,y.colorSpace),Ee=r.convert(y.type),ue=_(y.internalFormat,oe,Ee,y.normalized,y.colorSpace,y.isVideoTexture);Je(G,y);let le,we=y.mipmaps,De=y.isVideoTexture!==!0,Ne=ae.__version===void 0||W===!0,U=ne.dataReady,re=b(y,K);if(y.isDepthTexture)ue=T(y.format===Zi,y.type),Ne&&(De?n.texStorage2D(t.TEXTURE_2D,1,ue,K.width,K.height):n.texImage2D(t.TEXTURE_2D,0,ue,K.width,K.height,0,oe,Ee,null));else if(y.isDataTexture)if(we.length>0){De&&Ne&&n.texStorage2D(t.TEXTURE_2D,re,ue,we[0].width,we[0].height);for(let Q=0,ce=we.length;Q<ce;Q++)le=we[Q],De?U&&n.texSubImage2D(t.TEXTURE_2D,Q,0,0,le.width,le.height,oe,Ee,le.data):n.texImage2D(t.TEXTURE_2D,Q,ue,le.width,le.height,0,oe,Ee,le.data);y.generateMipmaps=!1}else De?(Ne&&n.texStorage2D(t.TEXTURE_2D,re,ue,K.width,K.height),U&&te(y,K,oe,Ee)):n.texImage2D(t.TEXTURE_2D,0,ue,K.width,K.height,0,oe,Ee,K.data);else if(y.isCompressedTexture)if(y.isCompressedArrayTexture){De&&Ne&&n.texStorage3D(t.TEXTURE_2D_ARRAY,re,ue,we[0].width,we[0].height,K.depth);for(let Q=0,ce=we.length;Q<ce;Q++)if(le=we[Q],y.format!==yi)if(oe!==null)if(De){if(U)if(y.layerUpdates.size>0){let me=Dg(le.width,le.height,y.format,y.type);for(let $ of y.layerUpdates){let Me=le.data.subarray($*me/le.data.BYTES_PER_ELEMENT,($+1)*me/le.data.BYTES_PER_ELEMENT);n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,Q,0,0,$,le.width,le.height,1,oe,Me)}y.clearLayerUpdates()}else n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,Q,0,0,0,le.width,le.height,K.depth,oe,le.data)}else n.compressedTexImage3D(t.TEXTURE_2D_ARRAY,Q,ue,le.width,le.height,K.depth,0,le.data,0,0);else Re("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else De?U&&n.texSubImage3D(t.TEXTURE_2D_ARRAY,Q,0,0,0,le.width,le.height,K.depth,oe,Ee,le.data):n.texImage3D(t.TEXTURE_2D_ARRAY,Q,ue,le.width,le.height,K.depth,0,oe,Ee,le.data)}else{De&&Ne&&n.texStorage2D(t.TEXTURE_2D,re,ue,we[0].width,we[0].height);for(let Q=0,ce=we.length;Q<ce;Q++)le=we[Q],y.format!==yi?oe!==null?De?U&&n.compressedTexSubImage2D(t.TEXTURE_2D,Q,0,0,le.width,le.height,oe,le.data):n.compressedTexImage2D(t.TEXTURE_2D,Q,ue,le.width,le.height,0,le.data):Re("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):De?U&&n.texSubImage2D(t.TEXTURE_2D,Q,0,0,le.width,le.height,oe,Ee,le.data):n.texImage2D(t.TEXTURE_2D,Q,ue,le.width,le.height,0,oe,Ee,le.data)}else if(y.isDataArrayTexture)if(De){if(Ne&&n.texStorage3D(t.TEXTURE_2D_ARRAY,re,ue,K.width,K.height,K.depth),U)if(y.layerUpdates.size>0){let Q=Dg(K.width,K.height,y.format,y.type);for(let ce of y.layerUpdates){let me=K.data.subarray(ce*Q/K.data.BYTES_PER_ELEMENT,(ce+1)*Q/K.data.BYTES_PER_ELEMENT);n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,ce,K.width,K.height,1,oe,Ee,me)}y.clearLayerUpdates()}else n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,0,K.width,K.height,K.depth,oe,Ee,K.data)}else n.texImage3D(t.TEXTURE_2D_ARRAY,0,ue,K.width,K.height,K.depth,0,oe,Ee,K.data);else if(y.isData3DTexture)De?(Ne&&n.texStorage3D(t.TEXTURE_3D,re,ue,K.width,K.height,K.depth),U&&n.texSubImage3D(t.TEXTURE_3D,0,0,0,0,K.width,K.height,K.depth,oe,Ee,K.data)):n.texImage3D(t.TEXTURE_3D,0,ue,K.width,K.height,K.depth,0,oe,Ee,K.data);else if(y.isFramebufferTexture){if(Ne)if(De)n.texStorage2D(t.TEXTURE_2D,re,ue,K.width,K.height);else{let Q=K.width,ce=K.height;for(let me=0;me<re;me++)n.texImage2D(t.TEXTURE_2D,me,ue,Q,ce,0,oe,Ee,null),Q>>=1,ce>>=1}}else if(y.isHTMLTexture){if("texElementImage2D"in t){let Q=t.canvas;if(Q.hasAttribute("layoutsubtree")||Q.setAttribute("layoutsubtree","true"),K.parentNode!==Q){Q.appendChild(K),p.add(y),Q.onpaint=ce=>{let me=ce.changedElements;for(let $ of p)me.includes($.image)&&($.needsUpdate=!0)},Q.requestPaint();return}if(t.texElementImage2D.length===3)t.texElementImage2D(t.TEXTURE_2D,t.RGBA8,K);else{let me=t.RGBA,$=t.RGBA,Me=t.UNSIGNED_BYTE;t.texElementImage2D(t.TEXTURE_2D,0,me,$,Me,K)}t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,t.LINEAR),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE)}}else if(we.length>0){if(De&&Ne){let Q=rt(we[0]);n.texStorage2D(t.TEXTURE_2D,re,ue,Q.width,Q.height)}for(let Q=0,ce=we.length;Q<ce;Q++)le=we[Q],De?U&&n.texSubImage2D(t.TEXTURE_2D,Q,0,0,oe,Ee,le):n.texImage2D(t.TEXTURE_2D,Q,ue,oe,Ee,le);y.generateMipmaps=!1}else if(De){if(Ne){let Q=rt(K);n.texStorage2D(t.TEXTURE_2D,re,ue,Q.width,Q.height)}U&&n.texSubImage2D(t.TEXTURE_2D,0,0,0,oe,Ee,K)}else n.texImage2D(t.TEXTURE_2D,0,ue,oe,Ee,K);f(y)&&g(G),ae.__version=ne.version,y.onUpdate&&y.onUpdate(y)}C.__version=y.version}function Ae(C,y,P){if(y.image.length!==6)return;let G=Z(C,y),W=y.source;n.bindTexture(t.TEXTURE_CUBE_MAP,C.__webglTexture,t.TEXTURE0+P);let ne=i.get(W);if(W.version!==ne.__version||G===!0){n.activeTexture(t.TEXTURE0+P);let ae=Ye.getPrimaries(Ye.workingColorSpace),X=y.colorSpace===ni?null:Ye.getPrimaries(y.colorSpace),K=y.colorSpace===ni||ae===X?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,y.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,y.premultiplyAlpha),n.pixelStorei(t.UNPACK_ALIGNMENT,y.unpackAlignment),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,K);let oe=y.isCompressedTexture||y.image[0].isCompressedTexture,Ee=y.image[0]&&y.image[0].isDataTexture,ue=[];for(let $=0;$<6;$++)!oe&&!Ee?ue[$]=m(y.image[$],!0,s.maxCubemapSize):ue[$]=Ee?y.image[$].image:y.image[$],ue[$]=Dn(y,ue[$]);let le=ue[0],we=r.convert(y.format,y.colorSpace),De=r.convert(y.type),Ne=_(y.internalFormat,we,De,y.normalized,y.colorSpace),U=y.isVideoTexture!==!0,re=ne.__version===void 0||G===!0,Q=W.dataReady,ce=b(y,le);Je(t.TEXTURE_CUBE_MAP,y);let me;if(oe){U&&re&&n.texStorage2D(t.TEXTURE_CUBE_MAP,ce,Ne,le.width,le.height);for(let $=0;$<6;$++){me=ue[$].mipmaps;for(let Me=0;Me<me.length;Me++){let ye=me[Me];y.format!==yi?we!==null?U?Q&&n.compressedTexSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me,0,0,ye.width,ye.height,we,ye.data):n.compressedTexImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me,Ne,ye.width,ye.height,0,ye.data):Re("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):U?Q&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me,0,0,ye.width,ye.height,we,De,ye.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me,Ne,ye.width,ye.height,0,we,De,ye.data)}}}else{if(me=y.mipmaps,U&&re){me.length>0&&ce++;let $=rt(ue[0]);n.texStorage2D(t.TEXTURE_CUBE_MAP,ce,Ne,$.width,$.height)}for(let $=0;$<6;$++)if(Ee){U?Q&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,0,0,ue[$].width,ue[$].height,we,De,ue[$].data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,Ne,ue[$].width,ue[$].height,0,we,De,ue[$].data);for(let Me=0;Me<me.length;Me++){let Bt=me[Me].image[$].image;U?Q&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me+1,0,0,Bt.width,Bt.height,we,De,Bt.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me+1,Ne,Bt.width,Bt.height,0,we,De,Bt.data)}}else{U?Q&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,0,0,we,De,ue[$]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,Ne,we,De,ue[$]);for(let Me=0;Me<me.length;Me++){let ye=me[Me];U?Q&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me+1,0,0,we,De,ye.image[$]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+$,Me+1,Ne,we,De,ye.image[$])}}}f(y)&&g(t.TEXTURE_CUBE_MAP),ne.__version=W.version,y.onUpdate&&y.onUpdate(y)}C.__version=y.version}function Ce(C,y,P,G,W,ne){let ae=r.convert(P.format,P.colorSpace),X=r.convert(P.type),K=_(P.internalFormat,ae,X,P.normalized,P.colorSpace),oe=i.get(y),Ee=i.get(P);if(Ee.__renderTarget=y,!oe.__hasExternalTextures){let ue=Math.max(1,y.width>>ne),le=Math.max(1,y.height>>ne);W===t.TEXTURE_3D||W===t.TEXTURE_2D_ARRAY?n.texImage3D(W,ne,K,ue,le,y.depth,0,ae,X,null):n.texImage2D(W,ne,K,ue,le,0,ae,X,null)}n.bindFramebuffer(t.FRAMEBUFFER,C),Wt(y)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,G,W,Ee.__webglTexture,0,Ut(y)):(W===t.TEXTURE_2D||W>=t.TEXTURE_CUBE_MAP_POSITIVE_X&&W<=t.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&t.framebufferTexture2D(t.FRAMEBUFFER,G,W,Ee.__webglTexture,ne),n.bindFramebuffer(t.FRAMEBUFFER,null)}function Rt(C,y,P){if(t.bindRenderbuffer(t.RENDERBUFFER,C),y.depthBuffer){let G=y.depthTexture,W=G&&G.isDepthTexture?G.type:null,ne=T(y.stencilBuffer,W),ae=y.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;Wt(y)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,Ut(y),ne,y.width,y.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,Ut(y),ne,y.width,y.height):t.renderbufferStorage(t.RENDERBUFFER,ne,y.width,y.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,ae,t.RENDERBUFFER,C)}else{let G=y.textures;for(let W=0;W<G.length;W++){let ne=G[W],ae=r.convert(ne.format,ne.colorSpace),X=r.convert(ne.type),K=_(ne.internalFormat,ae,X,ne.normalized,ne.colorSpace);Wt(y)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,Ut(y),K,y.width,y.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,Ut(y),K,y.width,y.height):t.renderbufferStorage(t.RENDERBUFFER,K,y.width,y.height)}}t.bindRenderbuffer(t.RENDERBUFFER,null)}function We(C,y,P){let G=y.isWebGLCubeRenderTarget===!0;if(n.bindFramebuffer(t.FRAMEBUFFER,C),!(y.depthTexture&&y.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let W=i.get(y.depthTexture);if(W.__renderTarget=y,(!W.__webglTexture||y.depthTexture.image.width!==y.width||y.depthTexture.image.height!==y.height)&&(y.depthTexture.image.width=y.width,y.depthTexture.image.height=y.height,y.depthTexture.needsUpdate=!0),G){if(W.__webglInit===void 0&&(W.__webglInit=!0,y.depthTexture.addEventListener("dispose",w)),W.__webglTexture===void 0){W.__webglTexture=t.createTexture(),n.bindTexture(t.TEXTURE_CUBE_MAP,W.__webglTexture),Je(t.TEXTURE_CUBE_MAP,y.depthTexture);let oe=r.convert(y.depthTexture.format),Ee=r.convert(y.depthTexture.type),ue;y.depthTexture.format===Yi?ue=t.DEPTH_COMPONENT24:y.depthTexture.format===Zi&&(ue=t.DEPTH24_STENCIL8);for(let le=0;le<6;le++)t.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,ue,y.width,y.height,0,oe,Ee,null)}}else j(y.depthTexture,0);let ne=W.__webglTexture,ae=Ut(y),X=G?t.TEXTURE_CUBE_MAP_POSITIVE_X+P:t.TEXTURE_2D,K=y.depthTexture.format===Zi?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;if(y.depthTexture.format===Yi)Wt(y)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,K,X,ne,0,ae):t.framebufferTexture2D(t.FRAMEBUFFER,K,X,ne,0);else if(y.depthTexture.format===Zi)Wt(y)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,K,X,ne,0,ae):t.framebufferTexture2D(t.FRAMEBUFFER,K,X,ne,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function ft(C){let y=i.get(C),P=C.isWebGLCubeRenderTarget===!0;if(y.__boundDepthTexture!==C.depthTexture){let G=C.depthTexture;if(y.__depthDisposeCallback&&y.__depthDisposeCallback(),G){let W=()=>{delete y.__boundDepthTexture,delete y.__depthDisposeCallback,G.removeEventListener("dispose",W)};G.addEventListener("dispose",W),y.__depthDisposeCallback=W}y.__boundDepthTexture=G}if(C.depthTexture&&!y.__autoAllocateDepthBuffer)if(P)for(let G=0;G<6;G++)We(y.__webglFramebuffer[G],C,G);else{let G=C.texture.mipmaps;G&&G.length>0?We(y.__webglFramebuffer[0],C,0):We(y.__webglFramebuffer,C,0)}else if(P){y.__webglDepthbuffer=[];for(let G=0;G<6;G++)if(n.bindFramebuffer(t.FRAMEBUFFER,y.__webglFramebuffer[G]),y.__webglDepthbuffer[G]===void 0)y.__webglDepthbuffer[G]=t.createRenderbuffer(),Rt(y.__webglDepthbuffer[G],C,!1);else{let W=C.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,ne=y.__webglDepthbuffer[G];t.bindRenderbuffer(t.RENDERBUFFER,ne),t.framebufferRenderbuffer(t.FRAMEBUFFER,W,t.RENDERBUFFER,ne)}}else{let G=C.texture.mipmaps;if(G&&G.length>0?n.bindFramebuffer(t.FRAMEBUFFER,y.__webglFramebuffer[0]):n.bindFramebuffer(t.FRAMEBUFFER,y.__webglFramebuffer),y.__webglDepthbuffer===void 0)y.__webglDepthbuffer=t.createRenderbuffer(),Rt(y.__webglDepthbuffer,C,!1);else{let W=C.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,ne=y.__webglDepthbuffer;t.bindRenderbuffer(t.RENDERBUFFER,ne),t.framebufferRenderbuffer(t.FRAMEBUFFER,W,t.RENDERBUFFER,ne)}}n.bindFramebuffer(t.FRAMEBUFFER,null)}function tt(C,y,P){let G=i.get(C);y!==void 0&&Ce(G.__webglFramebuffer,C,C.texture,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,0),P!==void 0&&ft(C)}function je(C){let y=C.texture,P=i.get(C),G=i.get(y);C.addEventListener("dispose",x);let W=C.textures,ne=C.isWebGLCubeRenderTarget===!0,ae=W.length>1;if(ae||(G.__webglTexture===void 0&&(G.__webglTexture=t.createTexture()),G.__version=y.version,a.memory.textures++),ne){P.__webglFramebuffer=[];for(let X=0;X<6;X++)if(y.mipmaps&&y.mipmaps.length>0){P.__webglFramebuffer[X]=[];for(let K=0;K<y.mipmaps.length;K++)P.__webglFramebuffer[X][K]=t.createFramebuffer()}else P.__webglFramebuffer[X]=t.createFramebuffer()}else{if(y.mipmaps&&y.mipmaps.length>0){P.__webglFramebuffer=[];for(let X=0;X<y.mipmaps.length;X++)P.__webglFramebuffer[X]=t.createFramebuffer()}else P.__webglFramebuffer=t.createFramebuffer();if(ae)for(let X=0,K=W.length;X<K;X++){let oe=i.get(W[X]);oe.__webglTexture===void 0&&(oe.__webglTexture=t.createTexture(),a.memory.textures++)}if(C.samples>0&&Wt(C)===!1){P.__webglMultisampledFramebuffer=t.createFramebuffer(),P.__webglColorRenderbuffer=[],n.bindFramebuffer(t.FRAMEBUFFER,P.__webglMultisampledFramebuffer);for(let X=0;X<W.length;X++){let K=W[X];P.__webglColorRenderbuffer[X]=t.createRenderbuffer(),t.bindRenderbuffer(t.RENDERBUFFER,P.__webglColorRenderbuffer[X]);let oe=r.convert(K.format,K.colorSpace),Ee=r.convert(K.type),ue=_(K.internalFormat,oe,Ee,K.normalized,K.colorSpace,C.isXRRenderTarget===!0),le=Ut(C);t.renderbufferStorageMultisample(t.RENDERBUFFER,le,ue,C.width,C.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+X,t.RENDERBUFFER,P.__webglColorRenderbuffer[X])}t.bindRenderbuffer(t.RENDERBUFFER,null),C.depthBuffer&&(P.__webglDepthRenderbuffer=t.createRenderbuffer(),Rt(P.__webglDepthRenderbuffer,C,!0)),n.bindFramebuffer(t.FRAMEBUFFER,null)}}if(ne){n.bindTexture(t.TEXTURE_CUBE_MAP,G.__webglTexture),Je(t.TEXTURE_CUBE_MAP,y);for(let X=0;X<6;X++)if(y.mipmaps&&y.mipmaps.length>0)for(let K=0;K<y.mipmaps.length;K++)Ce(P.__webglFramebuffer[X][K],C,y,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+X,K);else Ce(P.__webglFramebuffer[X],C,y,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+X,0);f(y)&&g(t.TEXTURE_CUBE_MAP),n.unbindTexture()}else if(ae){for(let X=0,K=W.length;X<K;X++){let oe=W[X],Ee=i.get(oe),ue=t.TEXTURE_2D;(C.isWebGL3DRenderTarget||C.isWebGLArrayRenderTarget)&&(ue=C.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(ue,Ee.__webglTexture),Je(ue,oe),Ce(P.__webglFramebuffer,C,oe,t.COLOR_ATTACHMENT0+X,ue,0),f(oe)&&g(ue)}n.unbindTexture()}else{let X=t.TEXTURE_2D;if((C.isWebGL3DRenderTarget||C.isWebGLArrayRenderTarget)&&(X=C.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(X,G.__webglTexture),Je(X,y),y.mipmaps&&y.mipmaps.length>0)for(let K=0;K<y.mipmaps.length;K++)Ce(P.__webglFramebuffer[K],C,y,t.COLOR_ATTACHMENT0,X,K);else Ce(P.__webglFramebuffer,C,y,t.COLOR_ATTACHMENT0,X,0);f(y)&&g(X),n.unbindTexture()}C.depthBuffer&&ft(C)}function kt(C){let y=C.textures;for(let P=0,G=y.length;P<G;P++){let W=y[P];if(f(W)){let ne=S(C),ae=i.get(W).__webglTexture;n.bindTexture(ne,ae),g(ne),n.unbindTexture()}}}let $t=[],rn=[];function fn(C){if(C.samples>0){if(Wt(C)===!1){let y=C.textures,P=C.width,G=C.height,W=t.COLOR_BUFFER_BIT,ne=C.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,ae=i.get(C),X=y.length>1;if(X)for(let oe=0;oe<y.length;oe++)n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.RENDERBUFFER,null),n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.TEXTURE_2D,null,0);n.bindFramebuffer(t.READ_FRAMEBUFFER,ae.__webglMultisampledFramebuffer);let K=C.texture.mipmaps;K&&K.length>0?n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ae.__webglFramebuffer[0]):n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ae.__webglFramebuffer);for(let oe=0;oe<y.length;oe++){if(C.resolveDepthBuffer&&(C.depthBuffer&&(W|=t.DEPTH_BUFFER_BIT),C.stencilBuffer&&C.resolveStencilBuffer&&(W|=t.STENCIL_BUFFER_BIT)),X){t.framebufferRenderbuffer(t.READ_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.RENDERBUFFER,ae.__webglColorRenderbuffer[oe]);let Ee=i.get(y[oe]).__webglTexture;t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,Ee,0)}t.blitFramebuffer(0,0,P,G,0,0,P,G,W,t.NEAREST),l===!0&&($t.length=0,rn.length=0,$t.push(t.COLOR_ATTACHMENT0+oe),C.depthBuffer&&C.resolveDepthBuffer===!1&&($t.push(ne),rn.push(ne),t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,rn)),t.invalidateFramebuffer(t.READ_FRAMEBUFFER,$t))}if(n.bindFramebuffer(t.READ_FRAMEBUFFER,null),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),X)for(let oe=0;oe<y.length;oe++){n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.RENDERBUFFER,ae.__webglColorRenderbuffer[oe]);let Ee=i.get(y[oe]).__webglTexture;n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.TEXTURE_2D,Ee,0)}n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ae.__webglMultisampledFramebuffer)}else if(C.depthBuffer&&C.resolveDepthBuffer===!1&&l){let y=C.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,[y])}}}function Ut(C){return Math.min(s.maxSamples,C.samples)}function Wt(C){let y=i.get(C);return C.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&y.__useRenderToTexture!==!1}function B(C){let y=a.render.frame;h.get(C)!==y&&(h.set(C,y),C.update())}function Dn(C,y){let P=C.colorSpace,G=C.format,W=C.type;return C.isCompressedTexture===!0||C.isVideoTexture===!0||P!==Ts&&P!==ni&&(Ye.getTransfer(P)===st?(G!==yi||W!==jt)&&Re("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Ue("WebGLTextures: Unsupported texture color space:",P)),y}function rt(C){return typeof HTMLImageElement<"u"&&C instanceof HTMLImageElement?(c.width=C.naturalWidth||C.width,c.height=C.naturalHeight||C.height):typeof VideoFrame<"u"&&C instanceof VideoFrame?(c.width=C.displayWidth,c.height=C.displayHeight):(c.width=C.width,c.height=C.height),c}this.allocateTextureUnit=k,this.resetTextureUnits=q,this.getTextureUnits=Y,this.setTextureUnits=N,this.setTexture2D=j,this.setTexture2DArray=ee,this.setTexture3D=se,this.setTextureCube=he,this.rebindTextures=tt,this.setupRenderTarget=je,this.updateRenderTargetMipmap=kt,this.updateMultisampleRenderTarget=fn,this.setupDepthRenderbuffer=ft,this.setupFrameBufferTexture=Ce,this.useMultisampledRTT=Wt,this.isReversedDepthBuffer=function(){return n.buffers.depth.getReversed()}}function fR(t,e){function n(i,s=ni){let r,a=Ye.getTransfer(s);if(i===jt)return t.UNSIGNED_BYTE;if(i===qf)return t.UNSIGNED_SHORT_4_4_4_4;if(i===Qf)return t.UNSIGNED_SHORT_5_5_5_1;if(i===_g)return t.UNSIGNED_INT_5_9_9_9_REV;if(i===Sg)return t.UNSIGNED_INT_10F_11F_11F_REV;if(i===xg)return t.BYTE;if(i===yg)return t.SHORT;if(i===Ao)return t.UNSIGNED_SHORT;if(i===Yf)return t.INT;if(i===Ui)return t.UNSIGNED_INT;if(i===ti)return t.FLOAT;if(i===Qi)return t.HALF_FLOAT;if(i===Ag)return t.ALPHA;if(i===Mg)return t.RGB;if(i===yi)return t.RGBA;if(i===Yi)return t.DEPTH_COMPONENT;if(i===Zi)return t.DEPTH_STENCIL;if(i===Eg)return t.RED;if(i===Zf)return t.RED_INTEGER;if(i===Sr)return t.RG;if(i===Kf)return t.RG_INTEGER;if(i===Jf)return t.RGBA_INTEGER;if(i===Jl||i===jl||i===$l||i===ec)if(a===st)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===Jl)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===jl)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===$l)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===ec)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===Jl)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===jl)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===$l)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===ec)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===jf||i===$f||i===eh||i===th)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===jf)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===$f)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===eh)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===th)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===nh||i===ih||i===sh||i===rh||i===ah||i===tc||i===oh)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===nh||i===ih)return a===st?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===sh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(i===rh)return r.COMPRESSED_R11_EAC;if(i===ah)return r.COMPRESSED_SIGNED_R11_EAC;if(i===tc)return r.COMPRESSED_RG11_EAC;if(i===oh)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===lh||i===ch||i===uh||i===fh||i===hh||i===dh||i===ph||i===mh||i===gh||i===vh||i===xh||i===yh||i===_h||i===Sh)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===lh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===ch)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===uh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===fh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===hh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===dh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===ph)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===mh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===gh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===vh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===xh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===yh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===_h)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===Sh)return a===st?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===Ah||i===Mh||i===Eh)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===Ah)return a===st?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===Mh)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===Eh)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===Th||i===bh||i===nc||i===wh)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===Th)return r.COMPRESSED_RED_RGTC1_EXT;if(i===bh)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===nc)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===wh)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===_r?t.UNSIGNED_INT_24_8:t[i]!==void 0?t[i]:null}return{convert:n}}var hR=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,dR=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,Zg=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,n){if(this.texture===null){let i=new Wl(e.texture);(e.depthNear!==n.depthNear||e.depthFar!==n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){let n=e.cameras[0].viewport,i=new Vt({vertexShader:hR,fragmentShader:dR,uniforms:{depthColor:{value:this.texture},depthWidth:{value:n.z},depthHeight:{value:n.w}}});this.mesh=new Mn(new ta(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},Kg=class extends zn{constructor(e,n){super();let i=this,s=null,r=1,a=null,o="local-floor",l=1,c=null,h=null,p=null,u=null,d=null,v=null,M=typeof XRWebGLBinding<"u",m=new Zg,f={},g=n.getContextAttributes(),S=null,_=null,T=[],b=[],w=new Ie,x=null,E=new Sn;E.viewport=new Dt;let R=new Sn;R.viewport=new Dt;let D=[E,R],L=new Hf,q=null,Y=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(Z){let ie=T[Z];return ie===void 0&&(ie=new xo,T[Z]=ie),ie.getTargetRaySpace()},this.getControllerGrip=function(Z){let ie=T[Z];return ie===void 0&&(ie=new xo,T[Z]=ie),ie.getGripSpace()},this.getHand=function(Z){let ie=T[Z];return ie===void 0&&(ie=new xo,T[Z]=ie),ie.getHandSpace()};function N(Z){let ie=b.indexOf(Z.inputSource);if(ie===-1)return;let te=T[ie];te!==void 0&&(te.update(Z.inputSource,Z.frame,c||a),te.dispatchEvent({type:Z.type,data:Z.inputSource}))}function k(){s.removeEventListener("select",N),s.removeEventListener("selectstart",N),s.removeEventListener("selectend",N),s.removeEventListener("squeeze",N),s.removeEventListener("squeezestart",N),s.removeEventListener("squeezeend",N),s.removeEventListener("end",k),s.removeEventListener("inputsourceschange",V);for(let Z=0;Z<T.length;Z++){let ie=b[Z];ie!==null&&(b[Z]=null,T[Z].disconnect(ie))}q=null,Y=null,m.reset();for(let Z in f)delete f[Z];e.setRenderTarget(S),d=null,u=null,p=null,s=null,_=null,Je.stop(),i.isPresenting=!1,e.setPixelRatio(x),e.setSize(w.width,w.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(Z){r=Z,i.isPresenting===!0&&Re("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(Z){o=Z,i.isPresenting===!0&&Re("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||a},this.setReferenceSpace=function(Z){c=Z},this.getBaseLayer=function(){return u!==null?u:d},this.getBinding=function(){return p===null&&M&&(p=new XRWebGLBinding(s,n)),p},this.getFrame=function(){return v},this.getSession=function(){return s},this.setSession=async function(Z){if(s=Z,s!==null){if(S=e.getRenderTarget(),s.addEventListener("select",N),s.addEventListener("selectstart",N),s.addEventListener("selectend",N),s.addEventListener("squeeze",N),s.addEventListener("squeezestart",N),s.addEventListener("squeezeend",N),s.addEventListener("end",k),s.addEventListener("inputsourceschange",V),g.xrCompatible!==!0&&await n.makeXRCompatible(),x=e.getPixelRatio(),e.getSize(w),M&&"createProjectionLayer"in XRWebGLBinding.prototype){let te=null,Se=null,Ae=null;g.depth&&(Ae=g.stencil?n.DEPTH24_STENCIL8:n.DEPTH_COMPONENT24,te=g.stencil?Zi:Yi,Se=g.stencil?_r:Ui);let Ce={colorFormat:n.RGBA8,depthFormat:Ae,scaleFactor:r};p=this.getBinding(),u=p.createProjectionLayer(Ce),s.updateRenderState({layers:[u]}),e.setPixelRatio(1),e.setSize(u.textureWidth,u.textureHeight,!1),_=new Nt(u.textureWidth,u.textureHeight,{format:yi,type:jt,depthTexture:new Ri(u.textureWidth,u.textureHeight,Se,void 0,void 0,void 0,void 0,void 0,void 0,te),stencilBuffer:g.stencil,colorSpace:e.outputColorSpace,samples:g.antialias?4:0,resolveDepthBuffer:u.ignoreDepthValues===!1,resolveStencilBuffer:u.ignoreDepthValues===!1})}else{let te={antialias:g.antialias,alpha:!0,depth:g.depth,stencil:g.stencil,framebufferScaleFactor:r};d=new XRWebGLLayer(s,n,te),s.updateRenderState({baseLayer:d}),e.setPixelRatio(1),e.setSize(d.framebufferWidth,d.framebufferHeight,!1),_=new Nt(d.framebufferWidth,d.framebufferHeight,{format:yi,type:jt,colorSpace:e.outputColorSpace,stencilBuffer:g.stencil,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}_.isXRRenderTarget=!0,this.setFoveation(l),c=null,a=await s.requestReferenceSpace(o),Je.setContext(s),Je.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return m.getDepthTexture()};function V(Z){for(let ie=0;ie<Z.removed.length;ie++){let te=Z.removed[ie],Se=b.indexOf(te);Se>=0&&(b[Se]=null,T[Se].disconnect(te))}for(let ie=0;ie<Z.added.length;ie++){let te=Z.added[ie],Se=b.indexOf(te);if(Se===-1){for(let Ce=0;Ce<T.length;Ce++)if(Ce>=b.length){b.push(te),Se=Ce;break}else if(b[Ce]===null){b[Ce]=te,Se=Ce;break}if(Se===-1)break}let Ae=T[Se];Ae&&Ae.connect(te)}}let j=new z,ee=new z;function se(Z,ie,te){j.setFromMatrixPosition(ie.matrixWorld),ee.setFromMatrixPosition(te.matrixWorld);let Se=j.distanceTo(ee),Ae=ie.projectionMatrix.elements,Ce=te.projectionMatrix.elements,Rt=Ae[14]/(Ae[10]-1),We=Ae[14]/(Ae[10]+1),ft=(Ae[9]+1)/Ae[5],tt=(Ae[9]-1)/Ae[5],je=(Ae[8]-1)/Ae[0],kt=(Ce[8]+1)/Ce[0],$t=Rt*je,rn=Rt*kt,fn=Se/(-je+kt),Ut=fn*-je;if(ie.matrixWorld.decompose(Z.position,Z.quaternion,Z.scale),Z.translateX(Ut),Z.translateZ(fn),Z.matrixWorld.compose(Z.position,Z.quaternion,Z.scale),Z.matrixWorldInverse.copy(Z.matrixWorld).invert(),Ae[10]===-1)Z.projectionMatrix.copy(ie.projectionMatrix),Z.projectionMatrixInverse.copy(ie.projectionMatrixInverse);else{let Wt=Rt+fn,B=We+fn,Dn=$t-Ut,rt=rn+(Se-Ut),C=ft*We/B*Wt,y=tt*We/B*Wt;Z.projectionMatrix.makePerspective(Dn,rt,C,y,Wt,B),Z.projectionMatrixInverse.copy(Z.projectionMatrix).invert()}}function he(Z,ie){ie===null?Z.matrixWorld.copy(Z.matrix):Z.matrixWorld.multiplyMatrices(ie.matrixWorld,Z.matrix),Z.matrixWorldInverse.copy(Z.matrixWorld).invert()}this.updateCamera=function(Z){if(s===null)return;let ie=Z.near,te=Z.far;m.texture!==null&&(m.depthNear>0&&(ie=m.depthNear),m.depthFar>0&&(te=m.depthFar)),L.near=R.near=E.near=ie,L.far=R.far=E.far=te,(q!==L.near||Y!==L.far)&&(s.updateRenderState({depthNear:L.near,depthFar:L.far}),q=L.near,Y=L.far),L.layers.mask=Z.layers.mask|6,E.layers.mask=L.layers.mask&-5,R.layers.mask=L.layers.mask&-3;let Se=Z.parent,Ae=L.cameras;he(L,Se);for(let Ce=0;Ce<Ae.length;Ce++)he(Ae[Ce],Se);Ae.length===2?se(L,E,R):L.projectionMatrix.copy(E.projectionMatrix),ve(Z,L,Se)};function ve(Z,ie,te){te===null?Z.matrix.copy(ie.matrixWorld):(Z.matrix.copy(te.matrixWorld),Z.matrix.invert(),Z.matrix.multiply(ie.matrixWorld)),Z.matrix.decompose(Z.position,Z.quaternion,Z.scale),Z.updateMatrixWorld(!0),Z.projectionMatrix.copy(ie.projectionMatrix),Z.projectionMatrixInverse.copy(ie.projectionMatrixInverse),Z.isPerspectiveCamera&&(Z.fov=_f*2*Math.atan(1/Z.projectionMatrix.elements[5]),Z.zoom=1)}this.getCamera=function(){return L},this.getFoveation=function(){if(!(u===null&&d===null))return l},this.setFoveation=function(Z){l=Z,u!==null&&(u.fixedFoveation=Z),d!==null&&d.fixedFoveation!==void 0&&(d.fixedFoveation=Z)},this.hasDepthSensing=function(){return m.texture!==null},this.getDepthSensingMesh=function(){return m.getMesh(L)},this.getCameraTexture=function(Z){return f[Z]};let Ke=null;function yt(Z,ie){if(h=ie.getViewerPose(c||a),v=ie,h!==null){let te=h.views;d!==null&&(e.setRenderTargetFramebuffer(_,d.framebuffer),e.setRenderTarget(_));let Se=!1;te.length!==L.cameras.length&&(L.cameras.length=0,Se=!0);for(let We=0;We<te.length;We++){let ft=te[We],tt=null;if(d!==null)tt=d.getViewport(ft);else{let kt=p.getViewSubImage(u,ft);tt=kt.viewport,We===0&&(e.setRenderTargetTextures(_,kt.colorTexture,kt.depthStencilTexture),e.setRenderTarget(_))}let je=D[We];je===void 0&&(je=new Sn,je.layers.enable(We),je.viewport=new Dt,D[We]=je),je.matrix.fromArray(ft.transform.matrix),je.matrix.decompose(je.position,je.quaternion,je.scale),je.projectionMatrix.fromArray(ft.projectionMatrix),je.projectionMatrixInverse.copy(je.projectionMatrix).invert(),je.viewport.set(tt.x,tt.y,tt.width,tt.height),We===0&&(L.matrix.copy(je.matrix),L.matrix.decompose(L.position,L.quaternion,L.scale)),Se===!0&&L.cameras.push(je)}let Ae=s.enabledFeatures;if(Ae&&Ae.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&M){p=i.getBinding();let We=p.getDepthInformation(te[0]);We&&We.isValid&&We.texture&&m.init(We,s.renderState)}if(Ae&&Ae.includes("camera-access")&&M){e.state.unbindTexture(),p=i.getBinding();for(let We=0;We<te.length;We++){let ft=te[We].camera;if(ft){let tt=f[ft];tt||(tt=new Wl,f[ft]=tt);let je=p.getCameraImage(ft);tt.sourceTexture=je}}}}for(let te=0;te<T.length;te++){let Se=b[te],Ae=T[te];Se!==null&&Ae!==void 0&&Ae.update(Se,ie,c||a)}Ke&&Ke(Z,ie),ie.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:ie}),v=null}let Je=new CA;Je.setAnimationLoop(yt),this.setAnimationLoop=function(Z){Ke=Z},this.dispose=function(){}}},pR=new Ht,PA=new Pe;PA.set(-1,0,0,0,1,0,0,0,1);function mR(t,e){function n(m,f){m.matrixAutoUpdate===!0&&m.updateMatrix(),f.value.copy(m.matrix)}function i(m,f){f.color.getRGB(m.fogColor.value,wg(t)),f.isFog?(m.fogNear.value=f.near,m.fogFar.value=f.far):f.isFogExp2&&(m.fogDensity.value=f.density)}function s(m,f,g,S,_){f.isNodeMaterial?f.uniformsNeedUpdate=!1:f.isMeshBasicMaterial?r(m,f):f.isMeshLambertMaterial?(r(m,f),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)):f.isMeshToonMaterial?(r(m,f),p(m,f)):f.isMeshPhongMaterial?(r(m,f),h(m,f),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)):f.isMeshStandardMaterial?(r(m,f),u(m,f),f.isMeshPhysicalMaterial&&d(m,f,_)):f.isMeshMatcapMaterial?(r(m,f),v(m,f)):f.isMeshDepthMaterial?r(m,f):f.isMeshDistanceMaterial?(r(m,f),M(m,f)):f.isMeshNormalMaterial?r(m,f):f.isLineBasicMaterial?(a(m,f),f.isLineDashedMaterial&&o(m,f)):f.isPointsMaterial?l(m,f,g,S):f.isSpriteMaterial?c(m,f):f.isShadowMaterial?(m.color.value.copy(f.color),m.opacity.value=f.opacity):f.isShaderMaterial&&(f.uniformsNeedUpdate=!1)}function r(m,f){m.opacity.value=f.opacity,f.color&&m.diffuse.value.copy(f.color),f.emissive&&m.emissive.value.copy(f.emissive).multiplyScalar(f.emissiveIntensity),f.map&&(m.map.value=f.map,n(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.bumpMap&&(m.bumpMap.value=f.bumpMap,n(f.bumpMap,m.bumpMapTransform),m.bumpScale.value=f.bumpScale,f.side===Jt&&(m.bumpScale.value*=-1)),f.normalMap&&(m.normalMap.value=f.normalMap,n(f.normalMap,m.normalMapTransform),m.normalScale.value.copy(f.normalScale),f.side===Jt&&m.normalScale.value.negate()),f.displacementMap&&(m.displacementMap.value=f.displacementMap,n(f.displacementMap,m.displacementMapTransform),m.displacementScale.value=f.displacementScale,m.displacementBias.value=f.displacementBias),f.emissiveMap&&(m.emissiveMap.value=f.emissiveMap,n(f.emissiveMap,m.emissiveMapTransform)),f.specularMap&&(m.specularMap.value=f.specularMap,n(f.specularMap,m.specularMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest);let g=e.get(f),S=g.envMap,_=g.envMapRotation;S&&(m.envMap.value=S,m.envMapRotation.value.setFromMatrix4(pR.makeRotationFromEuler(_)).transpose(),S.isCubeTexture&&S.isRenderTargetTexture===!1&&m.envMapRotation.value.premultiply(PA),m.reflectivity.value=f.reflectivity,m.ior.value=f.ior,m.refractionRatio.value=f.refractionRatio),f.lightMap&&(m.lightMap.value=f.lightMap,m.lightMapIntensity.value=f.lightMapIntensity,n(f.lightMap,m.lightMapTransform)),f.aoMap&&(m.aoMap.value=f.aoMap,m.aoMapIntensity.value=f.aoMapIntensity,n(f.aoMap,m.aoMapTransform))}function a(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,f.map&&(m.map.value=f.map,n(f.map,m.mapTransform))}function o(m,f){m.dashSize.value=f.dashSize,m.totalSize.value=f.dashSize+f.gapSize,m.scale.value=f.scale}function l(m,f,g,S){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.size.value=f.size*g,m.scale.value=S*.5,f.map&&(m.map.value=f.map,n(f.map,m.uvTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function c(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.rotation.value=f.rotation,f.map&&(m.map.value=f.map,n(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,n(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function h(m,f){m.specular.value.copy(f.specular),m.shininess.value=Math.max(f.shininess,1e-4)}function p(m,f){f.gradientMap&&(m.gradientMap.value=f.gradientMap)}function u(m,f){m.metalness.value=f.metalness,f.metalnessMap&&(m.metalnessMap.value=f.metalnessMap,n(f.metalnessMap,m.metalnessMapTransform)),m.roughness.value=f.roughness,f.roughnessMap&&(m.roughnessMap.value=f.roughnessMap,n(f.roughnessMap,m.roughnessMapTransform)),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)}function d(m,f,g){m.ior.value=f.ior,f.sheen>0&&(m.sheenColor.value.copy(f.sheenColor).multiplyScalar(f.sheen),m.sheenRoughness.value=f.sheenRoughness,f.sheenColorMap&&(m.sheenColorMap.value=f.sheenColorMap,n(f.sheenColorMap,m.sheenColorMapTransform)),f.sheenRoughnessMap&&(m.sheenRoughnessMap.value=f.sheenRoughnessMap,n(f.sheenRoughnessMap,m.sheenRoughnessMapTransform))),f.clearcoat>0&&(m.clearcoat.value=f.clearcoat,m.clearcoatRoughness.value=f.clearcoatRoughness,f.clearcoatMap&&(m.clearcoatMap.value=f.clearcoatMap,n(f.clearcoatMap,m.clearcoatMapTransform)),f.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=f.clearcoatRoughnessMap,n(f.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),f.clearcoatNormalMap&&(m.clearcoatNormalMap.value=f.clearcoatNormalMap,n(f.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(f.clearcoatNormalScale),f.side===Jt&&m.clearcoatNormalScale.value.negate())),f.dispersion>0&&(m.dispersion.value=f.dispersion),f.iridescence>0&&(m.iridescence.value=f.iridescence,m.iridescenceIOR.value=f.iridescenceIOR,m.iridescenceThicknessMinimum.value=f.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=f.iridescenceThicknessRange[1],f.iridescenceMap&&(m.iridescenceMap.value=f.iridescenceMap,n(f.iridescenceMap,m.iridescenceMapTransform)),f.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=f.iridescenceThicknessMap,n(f.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),f.transmission>0&&(m.transmission.value=f.transmission,m.transmissionSamplerMap.value=g.texture,m.transmissionSamplerSize.value.set(g.width,g.height),f.transmissionMap&&(m.transmissionMap.value=f.transmissionMap,n(f.transmissionMap,m.transmissionMapTransform)),m.thickness.value=f.thickness,f.thicknessMap&&(m.thicknessMap.value=f.thicknessMap,n(f.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=f.attenuationDistance,m.attenuationColor.value.copy(f.attenuationColor)),f.anisotropy>0&&(m.anisotropyVector.value.set(f.anisotropy*Math.cos(f.anisotropyRotation),f.anisotropy*Math.sin(f.anisotropyRotation)),f.anisotropyMap&&(m.anisotropyMap.value=f.anisotropyMap,n(f.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=f.specularIntensity,m.specularColor.value.copy(f.specularColor),f.specularColorMap&&(m.specularColorMap.value=f.specularColorMap,n(f.specularColorMap,m.specularColorMapTransform)),f.specularIntensityMap&&(m.specularIntensityMap.value=f.specularIntensityMap,n(f.specularIntensityMap,m.specularIntensityMapTransform))}function v(m,f){f.matcap&&(m.matcap.value=f.matcap)}function M(m,f){let g=e.get(f).light;m.referencePosition.value.setFromMatrixPosition(g.matrixWorld),m.nearDistance.value=g.shadow.camera.near,m.farDistance.value=g.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function gR(t,e,n,i){let s={},r={},a=[],o=t.getParameter(t.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,T){let b=T.program;i.uniformBlockBinding(_,b)}function c(_,T){let b=s[_.id];b===void 0&&(m(_),b=h(_),s[_.id]=b,_.addEventListener("dispose",g));let w=T.program;i.updateUBOMapping(_,w);let x=e.render.frame;r[_.id]!==x&&(u(_),r[_.id]=x)}function h(_){let T=p();_.__bindingPointIndex=T;let b=t.createBuffer(),w=_.__size,x=_.usage;return t.bindBuffer(t.UNIFORM_BUFFER,b),t.bufferData(t.UNIFORM_BUFFER,w,x),t.bindBuffer(t.UNIFORM_BUFFER,null),t.bindBufferBase(t.UNIFORM_BUFFER,T,b),b}function p(){for(let _=0;_<o;_++)if(a.indexOf(_)===-1)return a.push(_),_;return Ue("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function u(_){let T=s[_.id],b=_.uniforms,w=_.__cache;t.bindBuffer(t.UNIFORM_BUFFER,T);for(let x=0,E=b.length;x<E;x++){let R=b[x];if(Array.isArray(R))for(let D=0,L=R.length;D<L;D++)d(R[D],x,D,w);else d(R,x,0,w)}t.bindBuffer(t.UNIFORM_BUFFER,null)}function d(_,T,b,w){if(M(_,T,b,w)===!0){let x=_.__offset,E=_.value;if(Array.isArray(E)){let R=0;for(let D=0;D<E.length;D++){let L=E[D],q=f(L);v(L,_.__data,R),typeof L!="number"&&typeof L!="boolean"&&!L.isMatrix3&&!ArrayBuffer.isView(L)&&(R+=q.storage/Float32Array.BYTES_PER_ELEMENT)}}else v(E,_.__data,0);t.bufferSubData(t.UNIFORM_BUFFER,x,_.__data)}}function v(_,T,b){typeof _=="number"||typeof _=="boolean"?T[0]=_:_.isMatrix3?(T[0]=_.elements[0],T[1]=_.elements[1],T[2]=_.elements[2],T[3]=0,T[4]=_.elements[3],T[5]=_.elements[4],T[6]=_.elements[5],T[7]=0,T[8]=_.elements[6],T[9]=_.elements[7],T[10]=_.elements[8],T[11]=0):ArrayBuffer.isView(_)?T.set(new _.constructor(_.buffer,_.byteOffset,T.length)):_.toArray(T,b)}function M(_,T,b,w){let x=_.value,E=T+"_"+b;if(w[E]===void 0)return typeof x=="number"||typeof x=="boolean"?w[E]=x:ArrayBuffer.isView(x)?w[E]=x.slice():w[E]=x.clone(),!0;{let R=w[E];if(typeof x=="number"||typeof x=="boolean"){if(R!==x)return w[E]=x,!0}else{if(ArrayBuffer.isView(x))return!0;if(R.equals(x)===!1)return R.copy(x),!0}}return!1}function m(_){let T=_.uniforms,b=0,w=16;for(let E=0,R=T.length;E<R;E++){let D=Array.isArray(T[E])?T[E]:[T[E]];for(let L=0,q=D.length;L<q;L++){let Y=D[L],N=Array.isArray(Y.value)?Y.value:[Y.value];for(let k=0,V=N.length;k<V;k++){let j=N[k],ee=f(j),se=b%w,he=se%ee.boundary,ve=se+he;b+=he,ve!==0&&w-ve<ee.storage&&(b+=w-ve),Y.__data=new Float32Array(ee.storage/Float32Array.BYTES_PER_ELEMENT),Y.__offset=b,b+=ee.storage}}}let x=b%w;return x>0&&(b+=w-x),_.__size=b,_.__cache={},this}function f(_){let T={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(T.boundary=4,T.storage=4):_.isVector2?(T.boundary=8,T.storage=8):_.isVector3||_.isColor?(T.boundary=16,T.storage=12):_.isVector4?(T.boundary=16,T.storage=16):_.isMatrix3?(T.boundary=48,T.storage=48):_.isMatrix4?(T.boundary=64,T.storage=64):_.isTexture?Re("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(_)?(T.boundary=16,T.storage=_.byteLength):Re("WebGLRenderer: Unsupported uniform value type.",_),T}function g(_){let T=_.target;T.removeEventListener("dispose",g);let b=a.indexOf(T.__bindingPointIndex);a.splice(b,1),t.deleteBuffer(s[T.id]),delete s[T.id],delete r[T.id]}function S(){for(let _ in s)t.deleteBuffer(s[_]);a=[],s={},r={}}return{bind:l,update:c,dispose:S}}var vR=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),Ji=null;function xR(){return Ji===null&&(Ji=new Tf(vR,16,16,Sr,Qi),Ji.name="DFG_LUT",Ji.minFilter=xt,Ji.magFilter=xt,Ji.wrapS=Xi,Ji.wrapT=Xi,Ji.generateMipmaps=!1,Ji.needsUpdate=!0),Ji}var Ph=class{constructor(e={}){let{canvas:n=$S(),context:i=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:p=!1,reversedDepthBuffer:u=!1,outputBufferType:d=jt}=e;this.isWebGLRenderer=!0;let v;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");v=i.getContextAttributes().alpha}else v=a;let M=d,m=new Set([Jf,Kf,Zf]),f=new Set([jt,Ui,Ao,_r,qf,Qf]),g=new Uint32Array(4),S=new Int32Array(4),_=new z,T=null,b=null,w=[],x=[],E=null;this.domElement=n,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Di,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let R=this,D=!1,L=null,q=null,Y=null,N=null;this._outputColorSpace=Tt;let k=0,V=0,j=null,ee=-1,se=null,he=new Dt,ve=new Dt,Ke=null,yt=new ke(0),Je=0,Z=n.width,ie=n.height,te=1,Se=null,Ae=null,Ce=new Dt(0,0,Z,ie),Rt=new Dt(0,0,Z,ie),We=!1,ft=new Vl,tt=!1,je=!1,kt=new Ht,$t=new z,rn=new Dt,fn={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},Ut=!1;function Wt(){return j===null?te:1}let B=i;function Dn(A,I){return n.getContext(A,I)}try{let A={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:h,failIfMajorPerformanceCaveat:p};if("setAttribute"in n&&n.setAttribute("data-engine",`three.js r${"185"}`),n.addEventListener("webglcontextlost",Bt,!1),n.addEventListener("webglcontextrestored",mt,!1),n.addEventListener("webglcontextcreationerror",Bi,!1),B===null){let I="webgl2";if(B=Dn(I,A),B===null)throw Dn(I)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}}catch(A){throw Ue("WebGLRenderer: "+A.message),A}let rt,C,y,P,G,W,ne,ae,X,K,oe,Ee,ue,le,we,De,Ne,U,re,Q,ce,me,$;function Me(){rt=new TC(B),rt.init(),ce=new fR(B,rt),C=new vC(B,rt,e,ce),y=new cR(B,rt),C.reversedDepthBuffer&&u&&y.buffers.depth.setReversed(!0),q=B.createFramebuffer(),Y=B.createFramebuffer(),N=B.createFramebuffer(),P=new CC(B),G=new Z2,W=new uR(B,rt,y,G,C,ce,P),ne=new EC(R),ae=new BT(B),me=new mC(B,ae),X=new bC(B,ae,P,me),K=new DC(B,X,ae,me,P),U=new RC(B,C,W),we=new xC(G),oe=new Q2(R,ne,rt,C,me,we),Ee=new mR(R,G),ue=new J2,le=new iR(rt),Ne=new pC(R,ne,y,K,v,l),De=new lR(R,K,C),$=new gR(B,P,C,y),re=new gC(B,rt,P),Q=new wC(B,rt,P),P.programs=oe.programs,R.capabilities=C,R.extensions=rt,R.properties=G,R.renderLists=ue,R.shadowMap=De,R.state=y,R.info=P}Me(),M!==jt&&(E=new BC(M,n.width,n.height,o,s,r));let ye=new Kg(R,B);this.xr=ye,this.getContext=function(){return B},this.getContextAttributes=function(){return B.getContextAttributes()},this.forceContextLoss=function(){let A=rt.get("WEBGL_lose_context");A&&A.loseContext()},this.forceContextRestore=function(){let A=rt.get("WEBGL_lose_context");A&&A.restoreContext()},this.getPixelRatio=function(){return te},this.setPixelRatio=function(A){A!==void 0&&(te=A,this.setSize(Z,ie,!1))},this.getSize=function(A){return A.set(Z,ie)},this.setSize=function(A,I,H=!0){if(ye.isPresenting){Re("WebGLRenderer: Can't change size while VR device is presenting.");return}Z=A,ie=I,n.width=Math.floor(A*te),n.height=Math.floor(I*te),H===!0&&(n.style.width=A+"px",n.style.height=I+"px"),E!==null&&E.setSize(n.width,n.height),this.setViewport(0,0,A,I)},this.getDrawingBufferSize=function(A){return A.set(Z*te,ie*te).floor()},this.setDrawingBufferSize=function(A,I,H){Z=A,ie=I,te=H,n.width=Math.floor(A*H),n.height=Math.floor(I*H),this.setViewport(0,0,A,I)},this.setEffects=function(A){if(M===jt){Ue("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(A){for(let I=0;I<A.length;I++)if(A[I].isOutputPass===!0){Re("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}E.setEffects(A||[])},this.getCurrentViewport=function(A){return A.copy(he)},this.getViewport=function(A){return A.copy(Ce)},this.setViewport=function(A,I,H,O){A.isVector4?Ce.set(A.x,A.y,A.z,A.w):Ce.set(A,I,H,O),y.viewport(he.copy(Ce).multiplyScalar(te).round())},this.getScissor=function(A){return A.copy(Rt)},this.setScissor=function(A,I,H,O){A.isVector4?Rt.set(A.x,A.y,A.z,A.w):Rt.set(A,I,H,O),y.scissor(ve.copy(Rt).multiplyScalar(te).round())},this.getScissorTest=function(){return We},this.setScissorTest=function(A){y.setScissorTest(We=A)},this.setOpaqueSort=function(A){Se=A},this.setTransparentSort=function(A){Ae=A},this.getClearColor=function(A){return A.copy(Ne.getClearColor())},this.setClearColor=function(){Ne.setClearColor(...arguments)},this.getClearAlpha=function(){return Ne.getClearAlpha()},this.setClearAlpha=function(){Ne.setClearAlpha(...arguments)},this.clear=function(A=!0,I=!0,H=!0){let O=0;if(A){let F=!1;if(j!==null){let pe=j.texture.format;F=m.has(pe)}if(F){let pe=j.texture.type,xe=f.has(pe),de=Ne.getClearColor(),_e=Ne.getClearAlpha(),Te=de.r,Oe=de.g,Ge=de.b;xe?(g[0]=Te,g[1]=Oe,g[2]=Ge,g[3]=_e,B.clearBufferuiv(B.COLOR,0,g)):(S[0]=Te,S[1]=Oe,S[2]=Ge,S[3]=_e,B.clearBufferiv(B.COLOR,0,S))}else O|=B.COLOR_BUFFER_BIT}I&&(O|=B.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),H&&(O|=B.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),O!==0&&B.clear(O)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(A){A.setRenderer(this),L=A},this.dispose=function(){n.removeEventListener("webglcontextlost",Bt,!1),n.removeEventListener("webglcontextrestored",mt,!1),n.removeEventListener("webglcontextcreationerror",Bi,!1),Ne.dispose(),ue.dispose(),le.dispose(),G.dispose(),ne.dispose(),K.dispose(),me.dispose(),$.dispose(),oe.dispose(),ye.dispose(),ye.removeEventListener("sessionstart",rv),ye.removeEventListener("sessionend",av),Er.stop()};function Bt(A){A.preventDefault(),bg("WebGLRenderer: Context Lost."),D=!0}function mt(){bg("WebGLRenderer: Context Restored."),D=!1;let A=P.autoReset,I=De.enabled,H=De.autoUpdate,O=De.needsUpdate,F=De.type;Me(),P.autoReset=A,De.enabled=I,De.autoUpdate=H,De.needsUpdate=O,De.type=F}function Bi(A){Ue("WebGLRenderer: A WebGL context could not be created. Reason: ",A.statusMessage)}function Ii(A){let I=A.target;I.removeEventListener("dispose",Ii),ZA(I)}function ZA(A){KA(A),G.remove(A)}function KA(A){let I=G.get(A).programs;I!==void 0&&(I.forEach(function(H){oe.releaseProgram(H)}),A.isShaderMaterial&&oe.releaseShaderCache(A))}this.renderBufferDirect=function(A,I,H,O,F,pe){I===null&&(I=fn);let xe=F.isMesh&&F.matrixWorld.determinantAffine()<0,de=$A(A,I,H,O,F);y.setMaterial(O,xe);let _e=H.index,Te=1;if(O.wireframe===!0){if(_e=X.getWireframeAttribute(H),_e===void 0)return;Te=2}let Oe=H.drawRange,Ge=H.attributes.position,be=Oe.start*Te,lt=(Oe.start+Oe.count)*Te;pe!==null&&(be=Math.max(be,pe.start*Te),lt=Math.min(lt,(pe.start+pe.count)*Te)),_e!==null?(be=Math.max(be,0),lt=Math.min(lt,_e.count)):Ge!=null&&(be=Math.max(be,0),lt=Math.min(lt,Ge.count));let Ot=lt-be;if(Ot<0||Ot===1/0)return;me.setup(F,O,de,H,_e);let It,ht=re;if(_e!==null&&(It=ae.get(_e),ht=Q,ht.setIndex(It)),F.isMesh)O.wireframe===!0?(y.setLineWidth(O.wireframeLinewidth*Wt()),ht.setMode(B.LINES)):ht.setMode(B.TRIANGLES);else if(F.isLine){let vn=O.linewidth;vn===void 0&&(vn=1),y.setLineWidth(vn*Wt()),F.isLineSegments?ht.setMode(B.LINES):F.isLineLoop?ht.setMode(B.LINE_LOOP):ht.setMode(B.LINE_STRIP)}else F.isPoints?ht.setMode(B.POINTS):F.isSprite&&ht.setMode(B.TRIANGLES);if(F.isBatchedMesh)if(rt.get("WEBGL_multi_draw"))ht.renderMultiDraw(F._multiDrawStarts,F._multiDrawCounts,F._multiDrawCount);else{let vn=F._multiDrawStarts,ge=F._multiDrawCounts,Hn=F._multiDrawCount,$e=_e?ae.get(_e).bytesPerElement:1,ii=G.get(O).currentProgram.getUniforms();for(let Pi=0;Pi<Hn;Pi++)ii.setValue(B,"_gl_DrawID",Pi),ht.render(vn[Pi]/$e,ge[Pi])}else if(F.isInstancedMesh)ht.renderInstances(be,Ot,F.count);else if(H.isInstancedBufferGeometry){let vn=H._maxInstanceCount!==void 0?H._maxInstanceCount:1/0,ge=Math.min(H.instanceCount,vn);ht.renderInstances(be,Ot,ge)}else ht.render(be,Ot)};function sv(A,I,H){A.transparent===!0&&A.side===Rn&&A.forceSinglePass===!1?(A.side=Jt,A.needsUpdate=!0,uc(A,I,H),A.side=wi,A.needsUpdate=!0,uc(A,I,H),A.side=Rn):uc(A,I,H)}this.compile=function(A,I,H=null){H===null&&(H=A),b=le.get(H),b.init(I),x.push(b),H.traverseVisible(function(F){F.isLight&&F.layers.test(I.layers)&&(b.pushLight(F),F.castShadow&&b.pushShadow(F))}),A!==H&&A.traverseVisible(function(F){F.isLight&&F.layers.test(I.layers)&&(b.pushLight(F),F.castShadow&&b.pushShadow(F))}),b.setupLights();let O=new Set;return A.traverse(function(F){if(!(F.isMesh||F.isPoints||F.isLine||F.isSprite))return;let pe=F.material;if(pe)if(Array.isArray(pe))for(let xe=0;xe<pe.length;xe++){let de=pe[xe];sv(de,H,F),O.add(de)}else sv(pe,H,F),O.add(pe)}),b=x.pop(),O},this.compileAsync=function(A,I,H=null){let O=this.compile(A,I,H);return new Promise(F=>{function pe(){if(O.forEach(function(xe){G.get(xe).currentProgram.isReady()&&O.delete(xe)}),O.size===0){F(A);return}setTimeout(pe,10)}rt.get("KHR_parallel_shader_compile")!==null?pe():setTimeout(pe,10)})};let Hh=null;function JA(A){Hh&&Hh(A)}function rv(){Er.stop()}function av(){Er.start()}let Er=new CA;Er.setAnimationLoop(JA),typeof self<"u"&&Er.setContext(self),this.setAnimationLoop=function(A){Hh=A,ye.setAnimationLoop(A),A===null?Er.stop():Er.start()},ye.addEventListener("sessionstart",rv),ye.addEventListener("sessionend",av),this.render=function(A,I){if(I!==void 0&&I.isCamera!==!0){Ue("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(D===!0)return;L!==null&&L.renderStart(A,I);let H=ye.enabled===!0&&ye.isPresenting===!0,O=E!==null&&(j===null||H)&&E.begin(R,j);if(A.matrixWorldAutoUpdate===!0&&A.updateMatrixWorld(),I.parent===null&&I.matrixWorldAutoUpdate===!0&&I.updateMatrixWorld(),ye.enabled===!0&&ye.isPresenting===!0&&(E===null||E.isCompositing()===!1)&&(ye.cameraAutoUpdate===!0&&ye.updateCamera(I),I=ye.getCamera()),A.isScene===!0&&A.onBeforeRender(R,A,I,j),b=le.get(A,x.length),b.init(I),b.state.textureUnits=W.getTextureUnits(),x.push(b),kt.multiplyMatrices(I.projectionMatrix,I.matrixWorldInverse),ft.setFromProjectionMatrix(kt,bi,I.reversedDepth),je=this.localClippingEnabled,tt=we.init(this.clippingPlanes,je),T=ue.get(A,w.length),T.init(),w.push(T),ye.enabled===!0&&ye.isPresenting===!0){let xe=R.xr.getDepthSensingMesh();xe!==null&&Vh(xe,I,-1/0,R.sortObjects)}Vh(A,I,0,R.sortObjects),T.finish(),R.sortObjects===!0&&T.sort(Se,Ae,I.reversedDepth),Ut=ye.enabled===!1||ye.isPresenting===!1||ye.hasDepthSensing()===!1,Ut&&Ne.addToRenderList(T,A),this.info.render.frame++,this.info.autoReset===!0&&this.info.reset(),tt===!0&&we.beginShadows();let F=b.state.shadowsArray;if(De.render(F,A,I),tt===!0&&we.endShadows(),(O&&E.hasRenderPass())===!1){let xe=T.opaque,de=T.transmissive;if(b.setupLights(),I.isArrayCamera){let _e=I.cameras;if(de.length>0)for(let Te=0,Oe=_e.length;Te<Oe;Te++){let Ge=_e[Te];lv(xe,de,A,Ge)}Ut&&Ne.render(A);for(let Te=0,Oe=_e.length;Te<Oe;Te++){let Ge=_e[Te];ov(T,A,Ge,Ge.viewport)}}else de.length>0&&lv(xe,de,A,I),Ut&&Ne.render(A),ov(T,A,I)}j!==null&&V===0&&(W.updateMultisampleRenderTarget(j),W.updateRenderTargetMipmap(j)),O&&E.end(R),A.isScene===!0&&A.onAfterRender(R,A,I),me.resetDefaultState(),ee=-1,se=null,x.pop(),x.length>0?(b=x[x.length-1],W.setTextureUnits(b.state.textureUnits),tt===!0&&we.setGlobalState(R.clippingPlanes,b.state.camera)):b=null,w.pop(),w.length>0?T=w[w.length-1]:T=null,L!==null&&L.renderEnd()};function Vh(A,I,H,O){if(A.visible===!1)return;if(A.layers.test(I.layers)){if(A.isGroup)H=A.renderOrder;else if(A.isLOD)A.autoUpdate===!0&&A.update(I);else if(A.isLightProbeGrid)b.pushLightProbeGrid(A);else if(A.isLight)b.pushLight(A),A.castShadow&&b.pushShadow(A);else if(A.isSprite){if(!A.frustumCulled||ft.intersectsSprite(A)){O&&rn.setFromMatrixPosition(A.matrixWorld).applyMatrix4(kt);let xe=K.update(A),de=A.material;de.visible&&T.push(A,xe,de,H,rn.z,null)}}else if((A.isMesh||A.isLine||A.isPoints)&&(!A.frustumCulled||ft.intersectsObject(A))){let xe=K.update(A),de=A.material;if(O&&(A.boundingSphere!==void 0?(A.boundingSphere===null&&A.computeBoundingSphere(),rn.copy(A.boundingSphere.center)):(xe.boundingSphere===null&&xe.computeBoundingSphere(),rn.copy(xe.boundingSphere.center)),rn.applyMatrix4(A.matrixWorld).applyMatrix4(kt)),Array.isArray(de)){let _e=xe.groups;for(let Te=0,Oe=_e.length;Te<Oe;Te++){let Ge=_e[Te],be=de[Ge.materialIndex];be&&be.visible&&T.push(A,xe,be,H,rn.z,Ge)}}else de.visible&&T.push(A,xe,de,H,rn.z,null)}}let pe=A.children;for(let xe=0,de=pe.length;xe<de;xe++)Vh(pe[xe],I,H,O)}function ov(A,I,H,O){let{opaque:F,transmissive:pe,transparent:xe}=A;b.setupLightsView(H),tt===!0&&we.setGlobalState(R.clippingPlanes,H),O&&y.viewport(he.copy(O)),F.length>0&&cc(F,I,H),pe.length>0&&cc(pe,I,H),xe.length>0&&cc(xe,I,H),y.buffers.depth.setTest(!0),y.buffers.depth.setMask(!0),y.buffers.color.setMask(!0),y.setPolygonOffset(!1)}function lv(A,I,H,O){if((H.isScene===!0?H.overrideMaterial:null)!==null)return;if(b.state.transmissionRenderTarget[O.id]===void 0){let be=rt.has("EXT_color_buffer_half_float")||rt.has("EXT_color_buffer_float");b.state.transmissionRenderTarget[O.id]=new Nt(1,1,{generateMipmaps:!0,type:be?Qi:jt,minFilter:yr,samples:Math.max(4,C.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:Ye.workingColorSpace})}let pe=b.state.transmissionRenderTarget[O.id],xe=O.viewport||he;pe.setSize(xe.z*R.transmissionResolutionScale,xe.w*R.transmissionResolutionScale);let de=R.getRenderTarget(),_e=R.getActiveCubeFace(),Te=R.getActiveMipmapLevel();R.setRenderTarget(pe),R.getClearColor(yt),Je=R.getClearAlpha(),Je<1&&R.setClearColor(16777215,.5),R.clear(),Ut&&Ne.render(H);let Oe=R.toneMapping;R.toneMapping=Di;let Ge=O.viewport;if(O.viewport!==void 0&&(O.viewport=void 0),b.setupLightsView(O),tt===!0&&we.setGlobalState(R.clippingPlanes,O),cc(A,H,O),W.updateMultisampleRenderTarget(pe),W.updateRenderTargetMipmap(pe),rt.has("WEBGL_multisampled_render_to_texture")===!1){let be=!1;for(let lt=0,Ot=I.length;lt<Ot;lt++){let It=I[lt],{object:ht,geometry:vn,material:ge,group:Hn}=It;if(ge.side===Rn&&ht.layers.test(O.layers)){let $e=ge.side;ge.side=Jt,ge.needsUpdate=!0,cv(ht,H,O,vn,ge,Hn),ge.side=$e,ge.needsUpdate=!0,be=!0}}be===!0&&(W.updateMultisampleRenderTarget(pe),W.updateRenderTargetMipmap(pe))}R.setRenderTarget(de,_e,Te),R.setClearColor(yt,Je),Ge!==void 0&&(O.viewport=Ge),R.toneMapping=Oe}function cc(A,I,H){let O=I.isScene===!0?I.overrideMaterial:null;for(let F=0,pe=A.length;F<pe;F++){let xe=A[F],{object:de,geometry:_e,group:Te}=xe,Oe=xe.material;Oe.allowOverride===!0&&O!==null&&(Oe=O),de.layers.test(H.layers)&&cv(de,I,H,_e,Oe,Te)}}function cv(A,I,H,O,F,pe){A.onBeforeRender(R,I,H,O,F,pe),A.modelViewMatrix.multiplyMatrices(H.matrixWorldInverse,A.matrixWorld),A.normalMatrix.getNormalMatrix(A.modelViewMatrix),F.onBeforeRender(R,I,H,O,A,pe),F.transparent===!0&&F.side===Rn&&F.forceSinglePass===!1?(F.side=Jt,F.needsUpdate=!0,R.renderBufferDirect(H,I,O,F,A,pe),F.side=wi,F.needsUpdate=!0,R.renderBufferDirect(H,I,O,F,A,pe),F.side=Rn):R.renderBufferDirect(H,I,O,F,A,pe),A.onAfterRender(R,I,H,O,F,pe)}function uc(A,I,H){I.isScene!==!0&&(I=fn);let O=G.get(A),F=b.state.lights,pe=b.state.shadowsArray,xe=F.state.version,de=oe.getParameters(A,F.state,pe,I,H,b.state.lightProbeGridArray),_e=oe.getProgramCacheKey(de),Te=O.programs;O.environment=A.isMeshStandardMaterial||A.isMeshLambertMaterial||A.isMeshPhongMaterial?I.environment:null,O.fog=I.fog;let Oe=A.isMeshStandardMaterial||A.isMeshLambertMaterial&&!A.envMap||A.isMeshPhongMaterial&&!A.envMap;O.envMap=ne.get(A.envMap||O.environment,Oe),O.envMapRotation=O.environment!==null&&A.envMap===null?I.environmentRotation:A.envMapRotation,Te===void 0&&(A.addEventListener("dispose",Ii),Te=new Map,O.programs=Te);let Ge=Te.get(_e);if(Ge!==void 0){if(O.currentProgram===Ge&&O.lightsStateVersion===xe)return fv(A,de),Ge}else de.uniforms=oe.getUniforms(A),L!==null&&A.isNodeMaterial&&L.build(A,H,de),A.onBeforeCompile(de,R),Ge=oe.acquireProgram(de,_e),Te.set(_e,Ge),O.uniforms=de.uniforms;let be=O.uniforms;return(!A.isShaderMaterial&&!A.isRawShaderMaterial||A.clipping===!0)&&(be.clippingPlanes=we.uniform),fv(A,de),O.needsLights=t1(A),O.lightsStateVersion=xe,O.needsLights&&(be.ambientLightColor.value=F.state.ambient,be.lightProbe.value=F.state.probe,be.directionalLights.value=F.state.directional,be.directionalLightShadows.value=F.state.directionalShadow,be.spotLights.value=F.state.spot,be.spotLightShadows.value=F.state.spotShadow,be.rectAreaLights.value=F.state.rectArea,be.ltc_1.value=F.state.rectAreaLTC1,be.ltc_2.value=F.state.rectAreaLTC2,be.pointLights.value=F.state.point,be.pointLightShadows.value=F.state.pointShadow,be.hemisphereLights.value=F.state.hemi,be.directionalShadowMatrix.value=F.state.directionalShadowMatrix,be.spotLightMatrix.value=F.state.spotLightMatrix,be.spotLightMap.value=F.state.spotLightMap,be.pointShadowMatrix.value=F.state.pointShadowMatrix),O.lightProbeGrid=b.state.lightProbeGridArray.length>0,O.currentProgram=Ge,O.uniformsList=null,Ge}function uv(A){if(A.uniformsList===null){let I=A.currentProgram.getUniforms();A.uniformsList=Eo.seqWithValue(I.seq,A.uniforms)}return A.uniformsList}function fv(A,I){let H=G.get(A);H.outputColorSpace=I.outputColorSpace,H.batching=I.batching,H.batchingColor=I.batchingColor,H.instancing=I.instancing,H.instancingColor=I.instancingColor,H.instancingMorph=I.instancingMorph,H.skinning=I.skinning,H.morphTargets=I.morphTargets,H.morphNormals=I.morphNormals,H.morphColors=I.morphColors,H.morphTargetsCount=I.morphTargetsCount,H.numClippingPlanes=I.numClippingPlanes,H.numIntersection=I.numClipIntersection,H.vertexAlphas=I.vertexAlphas,H.vertexTangents=I.vertexTangents,H.toneMapping=I.toneMapping}function jA(A,I){if(A.length===0)return null;if(A.length===1)return A[0].texture!==null?A[0]:null;_.setFromMatrixPosition(I.matrixWorld);for(let H=0,O=A.length;H<O;H++){let F=A[H];if(F.texture!==null&&F.boundingBox.containsPoint(_))return F}return null}function $A(A,I,H,O,F){I.isScene!==!0&&(I=fn),W.resetTextureUnits();let pe=I.fog,xe=O.isMeshStandardMaterial||O.isMeshLambertMaterial||O.isMeshPhongMaterial?I.environment:null,de=j===null?R.outputColorSpace:j.isXRRenderTarget===!0?j.texture.colorSpace:Ye.workingColorSpace,_e=O.isMeshStandardMaterial||O.isMeshLambertMaterial&&!O.envMap||O.isMeshPhongMaterial&&!O.envMap,Te=ne.get(O.envMap||xe,_e),Oe=O.vertexColors===!0&&!!H.attributes.color&&H.attributes.color.itemSize===4,Ge=!!H.attributes.tangent&&(!!O.normalMap||O.anisotropy>0),be=!!H.morphAttributes.position,lt=!!H.morphAttributes.normal,Ot=!!H.morphAttributes.color,It=Di;O.toneMapped&&(j===null||j.isXRRenderTarget===!0)&&(It=R.toneMapping);let ht=H.morphAttributes.position||H.morphAttributes.normal||H.morphAttributes.color,vn=ht!==void 0?ht.length:0,ge=G.get(O),Hn=b.state.lights;if(tt===!0&&(je===!0||A!==se)){let gt=A===se&&O.id===ee;we.setState(O,A,gt)}let $e=!1;O.version===ge.__version?(ge.needsLights&&ge.lightsStateVersion!==Hn.state.version||ge.outputColorSpace!==de||F.isBatchedMesh&&ge.batching===!1||!F.isBatchedMesh&&ge.batching===!0||F.isBatchedMesh&&ge.batchingColor===!0&&F.colorTexture===null||F.isBatchedMesh&&ge.batchingColor===!1&&F.colorTexture!==null||F.isInstancedMesh&&ge.instancing===!1||!F.isInstancedMesh&&ge.instancing===!0||F.isSkinnedMesh&&ge.skinning===!1||!F.isSkinnedMesh&&ge.skinning===!0||F.isInstancedMesh&&ge.instancingColor===!0&&F.instanceColor===null||F.isInstancedMesh&&ge.instancingColor===!1&&F.instanceColor!==null||F.isInstancedMesh&&ge.instancingMorph===!0&&F.morphTexture===null||F.isInstancedMesh&&ge.instancingMorph===!1&&F.morphTexture!==null||ge.envMap!==Te||O.fog===!0&&ge.fog!==pe||ge.numClippingPlanes!==void 0&&(ge.numClippingPlanes!==we.numPlanes||ge.numIntersection!==we.numIntersection)||ge.vertexAlphas!==Oe||ge.vertexTangents!==Ge||ge.morphTargets!==be||ge.morphNormals!==lt||ge.morphColors!==Ot||ge.toneMapping!==It||ge.morphTargetsCount!==vn||!!ge.lightProbeGrid!=b.state.lightProbeGridArray.length>0)&&($e=!0):($e=!0,ge.__version=O.version);let ii=ge.currentProgram;$e===!0&&(ii=uc(O,I,F),L&&O.isNodeMaterial&&L.onUpdateProgram(O,ii,ge));let Pi=!1,ws=!1,oa=!1,dt=ii.getUniforms(),Ft=ge.uniforms;if(y.useProgram(ii.program)&&(Pi=!0,ws=!0,oa=!0),O.id!==ee&&(ee=O.id,ws=!0),ge.needsLights){let gt=jA(b.state.lightProbeGridArray,F);ge.lightProbeGrid!==gt&&(ge.lightProbeGrid=gt,ws=!0)}if(Pi||se!==A){y.buffers.depth.getReversed()&&A.reversedDepth!==!0&&(A._reversedDepth=!0,A.updateProjectionMatrix()),dt.setValue(B,"projectionMatrix",A.projectionMatrix),dt.setValue(B,"viewMatrix",A.matrixWorldInverse);let Rs=dt.map.cameraPosition;Rs!==void 0&&Rs.setValue(B,$t.setFromMatrixPosition(A.matrixWorld)),C.logarithmicDepthBuffer&&dt.setValue(B,"logDepthBufFC",2/(Math.log(A.far+1)/Math.LN2)),(O.isMeshPhongMaterial||O.isMeshToonMaterial||O.isMeshLambertMaterial||O.isMeshBasicMaterial||O.isMeshStandardMaterial||O.isShaderMaterial)&&dt.setValue(B,"isOrthographic",A.isOrthographicCamera===!0),se!==A&&(se=A,ws=!0,oa=!0)}if(ge.needsLights&&(Hn.state.directionalShadowMap.length>0&&dt.setValue(B,"directionalShadowMap",Hn.state.directionalShadowMap,W),Hn.state.spotShadowMap.length>0&&dt.setValue(B,"spotShadowMap",Hn.state.spotShadowMap,W),Hn.state.pointShadowMap.length>0&&dt.setValue(B,"pointShadowMap",Hn.state.pointShadowMap,W)),F.isSkinnedMesh){dt.setOptional(B,F,"bindMatrix"),dt.setOptional(B,F,"bindMatrixInverse");let gt=F.skeleton;gt&&(gt.boneTexture===null&&gt.computeBoneTexture(),dt.setValue(B,"boneTexture",gt.boneTexture,W))}F.isBatchedMesh&&(dt.setOptional(B,F,"batchingTexture"),dt.setValue(B,"batchingTexture",F._matricesTexture,W),dt.setOptional(B,F,"batchingIdTexture"),dt.setValue(B,"batchingIdTexture",F._indirectTexture,W),dt.setOptional(B,F,"batchingColorTexture"),F._colorsTexture!==null&&dt.setValue(B,"batchingColorTexture",F._colorsTexture,W));let Cs=H.morphAttributes;if((Cs.position!==void 0||Cs.normal!==void 0||Cs.color!==void 0)&&U.update(F,H,ii),(ws||ge.receiveShadow!==F.receiveShadow)&&(ge.receiveShadow=F.receiveShadow,dt.setValue(B,"receiveShadow",F.receiveShadow)),(O.isMeshStandardMaterial||O.isMeshLambertMaterial||O.isMeshPhongMaterial)&&O.envMap===null&&I.environment!==null&&(Ft.envMapIntensity.value=I.environmentIntensity),Ft.dfgLUT!==void 0&&(Ft.dfgLUT.value=xR()),ws){if(dt.setValue(B,"toneMappingExposure",R.toneMappingExposure),ge.needsLights&&e1(Ft,oa),pe&&O.fog===!0&&Ee.refreshFogUniforms(Ft,pe),Ee.refreshMaterialUniforms(Ft,O,te,ie,b.state.transmissionRenderTarget[A.id]),ge.needsLights&&ge.lightProbeGrid){let gt=ge.lightProbeGrid;Ft.probesSH.value=gt.texture,Ft.probesMin.value.copy(gt.boundingBox.min),Ft.probesMax.value.copy(gt.boundingBox.max),Ft.probesResolution.value.copy(gt.resolution)}Eo.upload(B,uv(ge),Ft,W)}if(O.isShaderMaterial&&O.uniformsNeedUpdate===!0&&(Eo.upload(B,uv(ge),Ft,W),O.uniformsNeedUpdate=!1),O.isSpriteMaterial&&dt.setValue(B,"center",F.center),dt.setValue(B,"modelViewMatrix",F.modelViewMatrix),dt.setValue(B,"normalMatrix",F.normalMatrix),dt.setValue(B,"modelMatrix",F.matrixWorld),O.uniformsGroups!==void 0){let gt=O.uniformsGroups;for(let Rs=0,la=gt.length;Rs<la;Rs++){let hv=gt[Rs];$.update(hv,ii),$.bind(hv,ii)}}return ii}function e1(A,I){A.ambientLightColor.needsUpdate=I,A.lightProbe.needsUpdate=I,A.directionalLights.needsUpdate=I,A.directionalLightShadows.needsUpdate=I,A.pointLights.needsUpdate=I,A.pointLightShadows.needsUpdate=I,A.spotLights.needsUpdate=I,A.spotLightShadows.needsUpdate=I,A.rectAreaLights.needsUpdate=I,A.hemisphereLights.needsUpdate=I}function t1(A){return A.isMeshLambertMaterial||A.isMeshToonMaterial||A.isMeshPhongMaterial||A.isMeshStandardMaterial||A.isShadowMaterial||A.isShaderMaterial&&A.lights===!0}this.getActiveCubeFace=function(){return k},this.getActiveMipmapLevel=function(){return V},this.getRenderTarget=function(){return j},this.setRenderTargetTextures=function(A,I,H){let O=G.get(A);O.__autoAllocateDepthBuffer=A.resolveDepthBuffer===!1,O.__autoAllocateDepthBuffer===!1&&(O.__useRenderToTexture=!1),G.get(A.texture).__webglTexture=I,G.get(A.depthTexture).__webglTexture=O.__autoAllocateDepthBuffer?void 0:H,O.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(A,I){let H=G.get(A);H.__webglFramebuffer=I,H.__useDefaultFramebuffer=I===void 0},this.setRenderTarget=function(A,I=0,H=0){j=A,k=I,V=H;let O=null,F=!1,pe=!1;if(A){let de=G.get(A);if(de.__useDefaultFramebuffer!==void 0){y.bindFramebuffer(B.FRAMEBUFFER,de.__webglFramebuffer),he.copy(A.viewport),ve.copy(A.scissor),Ke=A.scissorTest,y.viewport(he),y.scissor(ve),y.setScissorTest(Ke),ee=-1;return}else if(de.__webglFramebuffer===void 0)W.setupRenderTarget(A);else if(de.__hasExternalTextures)W.rebindTextures(A,G.get(A.texture).__webglTexture,G.get(A.depthTexture).__webglTexture);else if(A.depthBuffer){let Oe=A.depthTexture;if(de.__boundDepthTexture!==Oe){if(Oe!==null&&G.has(Oe)&&(A.width!==Oe.image.width||A.height!==Oe.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");W.setupDepthRenderbuffer(A)}}let _e=A.texture;(_e.isData3DTexture||_e.isDataArrayTexture||_e.isCompressedArrayTexture)&&(pe=!0);let Te=G.get(A).__webglFramebuffer;A.isWebGLCubeRenderTarget?(Array.isArray(Te[I])?O=Te[I][H]:O=Te[I],F=!0):A.samples>0&&W.useMultisampledRTT(A)===!1?O=G.get(A).__webglMultisampledFramebuffer:Array.isArray(Te)?O=Te[H]:O=Te,he.copy(A.viewport),ve.copy(A.scissor),Ke=A.scissorTest}else he.copy(Ce).multiplyScalar(te).floor(),ve.copy(Rt).multiplyScalar(te).floor(),Ke=We;if(H!==0&&(O=q),y.bindFramebuffer(B.FRAMEBUFFER,O)&&y.drawBuffers(A,O),y.viewport(he),y.scissor(ve),y.setScissorTest(Ke),F){let de=G.get(A.texture);B.framebufferTexture2D(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_CUBE_MAP_POSITIVE_X+I,de.__webglTexture,H)}else if(pe){let de=I;for(let _e=0;_e<A.textures.length;_e++){let Te=G.get(A.textures[_e]);B.framebufferTextureLayer(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0+_e,Te.__webglTexture,H,de)}}else if(A!==null&&H!==0){let de=G.get(A.texture);B.framebufferTexture2D(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,de.__webglTexture,H)}ee=-1},this.readRenderTargetPixels=function(A,I,H,O,F,pe,xe,de=0){if(!(A&&A.isWebGLRenderTarget)){Ue("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let _e=G.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&xe!==void 0&&(_e=_e[xe]),_e){y.bindFramebuffer(B.FRAMEBUFFER,_e);try{let Te=A.textures[de],Oe=Te.format,Ge=Te.type;if(A.textures.length>1&&B.readBuffer(B.COLOR_ATTACHMENT0+de),!C.textureFormatReadable(Oe)){Ue("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!C.textureTypeReadable(Ge)){Ue("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}I>=0&&I<=A.width-O&&H>=0&&H<=A.height-F&&B.readPixels(I,H,O,F,ce.convert(Oe),ce.convert(Ge),pe)}finally{let Te=j!==null?G.get(j).__webglFramebuffer:null;y.bindFramebuffer(B.FRAMEBUFFER,Te)}}},this.readRenderTargetPixelsAsync=async function(A,I,H,O,F,pe,xe,de=0){if(!(A&&A.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let _e=G.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&xe!==void 0&&(_e=_e[xe]),_e)if(I>=0&&I<=A.width-O&&H>=0&&H<=A.height-F){y.bindFramebuffer(B.FRAMEBUFFER,_e);let Te=A.textures[de],Oe=Te.format,Ge=Te.type;if(A.textures.length>1&&B.readBuffer(B.COLOR_ATTACHMENT0+de),!C.textureFormatReadable(Oe))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!C.textureTypeReadable(Ge))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let be=B.createBuffer();B.bindBuffer(B.PIXEL_PACK_BUFFER,be),B.bufferData(B.PIXEL_PACK_BUFFER,pe.byteLength,B.STREAM_READ),B.readPixels(I,H,O,F,ce.convert(Oe),ce.convert(Ge),0);let lt=j!==null?G.get(j).__webglFramebuffer:null;y.bindFramebuffer(B.FRAMEBUFFER,lt);let Ot=B.fenceSync(B.SYNC_GPU_COMMANDS_COMPLETE,0);return B.flush(),await tA(B,Ot,4),B.bindBuffer(B.PIXEL_PACK_BUFFER,be),B.getBufferSubData(B.PIXEL_PACK_BUFFER,0,pe),B.deleteBuffer(be),B.deleteSync(Ot),pe}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(A,I=null,H=0){let O=Math.pow(2,-H),F=Math.floor(A.image.width*O),pe=Math.floor(A.image.height*O),xe=I!==null?I.x:0,de=I!==null?I.y:0;W.setTexture2D(A,0),B.copyTexSubImage2D(B.TEXTURE_2D,H,0,0,xe,de,F,pe),y.unbindTexture()},this.copyTextureToTexture=function(A,I,H=null,O=null,F=0,pe=0){let xe,de,_e,Te,Oe,Ge,be,lt,Ot,It=A.isCompressedTexture?A.mipmaps[pe]:A.image;if(H!==null)xe=H.max.x-H.min.x,de=H.max.y-H.min.y,_e=H.isBox3?H.max.z-H.min.z:1,Te=H.min.x,Oe=H.min.y,Ge=H.isBox3?H.min.z:0;else{let Ft=Math.pow(2,-F);xe=Math.floor(It.width*Ft),de=Math.floor(It.height*Ft),A.isDataArrayTexture?_e=It.depth:A.isData3DTexture?_e=Math.floor(It.depth*Ft):_e=1,Te=0,Oe=0,Ge=0}O!==null?(be=O.x,lt=O.y,Ot=O.z):(be=0,lt=0,Ot=0);let ht=ce.convert(I.format),vn=ce.convert(I.type),ge;I.isData3DTexture?(W.setTexture3D(I,0),ge=B.TEXTURE_3D):I.isDataArrayTexture||I.isCompressedArrayTexture?(W.setTexture2DArray(I,0),ge=B.TEXTURE_2D_ARRAY):(W.setTexture2D(I,0),ge=B.TEXTURE_2D),y.activeTexture(B.TEXTURE0),y.pixelStorei(B.UNPACK_FLIP_Y_WEBGL,I.flipY),y.pixelStorei(B.UNPACK_PREMULTIPLY_ALPHA_WEBGL,I.premultiplyAlpha),y.pixelStorei(B.UNPACK_ALIGNMENT,I.unpackAlignment);let Hn=y.getParameter(B.UNPACK_ROW_LENGTH),$e=y.getParameter(B.UNPACK_IMAGE_HEIGHT),ii=y.getParameter(B.UNPACK_SKIP_PIXELS),Pi=y.getParameter(B.UNPACK_SKIP_ROWS),ws=y.getParameter(B.UNPACK_SKIP_IMAGES);y.pixelStorei(B.UNPACK_ROW_LENGTH,It.width),y.pixelStorei(B.UNPACK_IMAGE_HEIGHT,It.height),y.pixelStorei(B.UNPACK_SKIP_PIXELS,Te),y.pixelStorei(B.UNPACK_SKIP_ROWS,Oe),y.pixelStorei(B.UNPACK_SKIP_IMAGES,Ge);let oa=A.isDataArrayTexture||A.isData3DTexture,dt=I.isDataArrayTexture||I.isData3DTexture;if(A.isDepthTexture){let Ft=G.get(A),Cs=G.get(I),gt=G.get(Ft.__renderTarget),Rs=G.get(Cs.__renderTarget);y.bindFramebuffer(B.READ_FRAMEBUFFER,gt.__webglFramebuffer),y.bindFramebuffer(B.DRAW_FRAMEBUFFER,Rs.__webglFramebuffer);for(let la=0;la<_e;la++)oa&&(B.framebufferTextureLayer(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,G.get(A).__webglTexture,F,Ge+la),B.framebufferTextureLayer(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,G.get(I).__webglTexture,pe,Ot+la)),B.blitFramebuffer(Te,Oe,xe,de,be,lt,xe,de,B.DEPTH_BUFFER_BIT,B.NEAREST);y.bindFramebuffer(B.READ_FRAMEBUFFER,null),y.bindFramebuffer(B.DRAW_FRAMEBUFFER,null)}else if(F!==0||A.isRenderTargetTexture||G.has(A)){let Ft=G.get(A),Cs=G.get(I);y.bindFramebuffer(B.READ_FRAMEBUFFER,Y),y.bindFramebuffer(B.DRAW_FRAMEBUFFER,N);for(let gt=0;gt<_e;gt++)oa?B.framebufferTextureLayer(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,Ft.__webglTexture,F,Ge+gt):B.framebufferTexture2D(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,Ft.__webglTexture,F),dt?B.framebufferTextureLayer(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,Cs.__webglTexture,pe,Ot+gt):B.framebufferTexture2D(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,Cs.__webglTexture,pe),F!==0?B.blitFramebuffer(Te,Oe,xe,de,be,lt,xe,de,B.COLOR_BUFFER_BIT,B.NEAREST):dt?B.copyTexSubImage3D(ge,pe,be,lt,Ot+gt,Te,Oe,xe,de):B.copyTexSubImage2D(ge,pe,be,lt,Te,Oe,xe,de);y.bindFramebuffer(B.READ_FRAMEBUFFER,null),y.bindFramebuffer(B.DRAW_FRAMEBUFFER,null)}else dt?A.isDataTexture||A.isData3DTexture?B.texSubImage3D(ge,pe,be,lt,Ot,xe,de,_e,ht,vn,It.data):I.isCompressedArrayTexture?B.compressedTexSubImage3D(ge,pe,be,lt,Ot,xe,de,_e,ht,It.data):B.texSubImage3D(ge,pe,be,lt,Ot,xe,de,_e,ht,vn,It):A.isDataTexture?B.texSubImage2D(B.TEXTURE_2D,pe,be,lt,xe,de,ht,vn,It.data):A.isCompressedTexture?B.compressedTexSubImage2D(B.TEXTURE_2D,pe,be,lt,It.width,It.height,ht,It.data):B.texSubImage2D(B.TEXTURE_2D,pe,be,lt,xe,de,ht,vn,It);y.pixelStorei(B.UNPACK_ROW_LENGTH,Hn),y.pixelStorei(B.UNPACK_IMAGE_HEIGHT,$e),y.pixelStorei(B.UNPACK_SKIP_PIXELS,ii),y.pixelStorei(B.UNPACK_SKIP_ROWS,Pi),y.pixelStorei(B.UNPACK_SKIP_IMAGES,ws),pe===0&&I.generateMipmaps&&B.generateMipmap(ge),y.unbindTexture()},this.initRenderTarget=function(A){G.get(A).__webglFramebuffer===void 0&&W.setupRenderTarget(A)},this.initTexture=function(A){A.isCubeTexture?W.setTextureCube(A,0):A.isData3DTexture?W.setTexture3D(A,0):A.isDataArrayTexture||A.isCompressedArrayTexture?W.setTexture2DArray(A,0):W.setTexture2D(A,0),y.unbindTexture()},this.resetState=function(){k=0,V=0,j=null,y.reset(),me.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return bi}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let n=this.getContext();n.drawingBufferColorSpace=Ye._getDrawingBufferColorSpace(e),n.unpackColorSpace=Ye._getUnpackColorSpace()}};var _R=(()=>{let t=new Float32Array([-1,-1,0,3,-1,0,-1,3,0]),e=new Float32Array([0,0,2,0,0,2]),n=new xi;return n.setAttribute("position",new An(t,3)),n.setAttribute("uv",new An(e,2)),n})(),Mr=class ev{static get fullscreenGeometry(){return _R}constructor(e="Pass",n=new dr,i=new bs){this.name=e,this.renderer=null,this.scene=n,this.camera=i,this.screen=null,this.rtt=!0,this.needsSwap=!0,this.needsDepthBlit=!1,this.needsDepthTexture=!1,this.enabled=!0}get renderToScreen(){return!this.rtt}set renderToScreen(e){if(this.rtt===e){let n=this.fullscreenMaterial;n!==null&&(n.needsUpdate=!0),this.rtt=!e}}set mainScene(e){}set mainCamera(e){}setRenderer(e){this.renderer=e}isEnabled(){return this.enabled}setEnabled(e){this.enabled=e}get fullscreenMaterial(){return this.screen!==null?this.screen.material:null}set fullscreenMaterial(e){let n=this.screen;n!==null?n.material=e:(n=new Mn(ev.fullscreenGeometry,e),n.frustumCulled=!1,this.scene===null&&(this.scene=new dr),this.scene.add(n),this.screen=n)}getFullscreenMaterial(){return this.fullscreenMaterial}setFullscreenMaterial(e){this.fullscreenMaterial=e}getDepthTexture(){return null}setDepthTexture(e,n=Ki){}render(e,n,i,s,r){throw new Error("Render method not implemented!")}setSize(e,n){}initialize(e,n,i){}dispose(){for(let e of Object.keys(this)){let n=this[e];(n instanceof Nt||n instanceof Ci||n instanceof Kt||n instanceof ev)&&this[e].dispose()}this.fullscreenMaterial!==null&&this.fullscreenMaterial.dispose()}},SR=class extends Mr{constructor(){super("ClearMaskPass",null,null),this.needsSwap=!1}render(t,e,n,i,s){let r=t.state.buffers.stencil;r.setLocked(!1),r.setTest(!1)}},AR=`#ifdef COLOR_WRITE
#include <common>
#include <dithering_pars_fragment>
#ifdef FRAMEBUFFER_PRECISION_HIGH
uniform mediump sampler2D inputBuffer;
#else
uniform lowp sampler2D inputBuffer;
#endif
#endif
#ifdef DEPTH_WRITE
#include <packing>
#ifdef GL_FRAGMENT_PRECISION_HIGH
uniform highp sampler2D depthBuffer;
#else
uniform mediump sampler2D depthBuffer;
#endif
float readDepth(const in vec2 uv){
#if DEPTH_PACKING == 3201
return unpackRGBAToDepth(texture2D(depthBuffer,uv));
#else
return texture2D(depthBuffer,uv).r;
#endif
}
#endif
#ifdef USE_WEIGHTS
uniform vec4 channelWeights;
#endif
uniform float opacity;varying vec2 vUv;void main(){
#ifdef COLOR_WRITE
vec4 texel=texture2D(inputBuffer,vUv);
#ifdef USE_WEIGHTS
texel*=channelWeights;
#endif
gl_FragColor=opacity*texel;
#ifdef COLOR_SPACE_CONVERSION
#include <colorspace_fragment>
#endif
#include <dithering_fragment>
#else
gl_FragColor=vec4(0.0);
#endif
#ifdef DEPTH_WRITE
gl_FragDepth=readDepth(vUv);
#endif
}`,MR="varying vec2 vUv;void main(){vUv=position.xy*0.5+0.5;gl_Position=vec4(position.xy,1.0,1.0);}",ER=class extends Vt{constructor(){super({name:"CopyMaterial",defines:{COLOR_SPACE_CONVERSION:"1",DEPTH_PACKING:"0",COLOR_WRITE:"1"},uniforms:{inputBuffer:new Ct(null),depthBuffer:new Ct(null),channelWeights:new Ct(null),opacity:new Ct(1)},blending:Gn,toneMapped:!1,depthWrite:!1,depthTest:!1,fragmentShader:AR,vertexShader:MR}),this.depthFunc=mo}get inputBuffer(){return this.uniforms.inputBuffer.value}set inputBuffer(t){let e=t!==null;this.colorWrite!==e&&(e?this.defines.COLOR_WRITE=!0:delete this.defines.COLOR_WRITE,this.colorWrite=e,this.needsUpdate=!0),this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){let e=t!==null;this.depthWrite!==e&&(e?this.defines.DEPTH_WRITE=!0:delete this.defines.DEPTH_WRITE,this.depthTest=e,this.depthWrite=e,this.needsUpdate=!0),this.uniforms.depthBuffer.value=t}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}get colorSpaceConversion(){return this.defines.COLOR_SPACE_CONVERSION!==void 0}set colorSpaceConversion(t){this.colorSpaceConversion!==t&&(t?this.defines.COLOR_SPACE_CONVERSION=!0:delete this.defines.COLOR_SPACE_CONVERSION,this.needsUpdate=!0)}get channelWeights(){return this.uniforms.channelWeights.value}set channelWeights(t){t!==null?(this.defines.USE_WEIGHTS="1",this.uniforms.channelWeights.value=t):delete this.defines.USE_WEIGHTS,this.needsUpdate=!0}setInputBuffer(t){this.uniforms.inputBuffer.value=t}getOpacity(t){return this.uniforms.opacity.value}setOpacity(t){this.uniforms.opacity.value=t}},TR=class extends Mr{constructor(t,e=!0){super("CopyPass"),this.fullscreenMaterial=new ER,this.needsSwap=!1,this.renderTarget=t,t===void 0&&(this.renderTarget=new Nt(1,1,{minFilter:xt,magFilter:xt,stencilBuffer:!1,depthBuffer:!1}),this.renderTarget.texture.name="CopyPass.Target"),this.autoResize=e}get resize(){return this.autoResize}set resize(t){this.autoResize=t}get texture(){return this.renderTarget.texture}getTexture(){return this.renderTarget.texture}setAutoResizeEnabled(t){this.autoResize=t}render(t,e,n,i,s){this.fullscreenMaterial.inputBuffer=e.texture,t.setRenderTarget(this.renderToScreen?null:this.renderTarget),t.render(this.scene,this.camera)}setSize(t,e){this.autoResize&&this.renderTarget.setSize(t,e)}initialize(t,e,n){n!==void 0&&(this.renderTarget.texture.type=n,n!==jt?this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1":t!==null&&t.outputColorSpace===Tt&&(this.renderTarget.texture.colorSpace=Tt))}},LA=new ke,FA=class extends Mr{constructor(t=!0,e=!0,n=!1){super("ClearPass",null,null),this.needsSwap=!1,this.color=t,this.depth=e,this.stencil=n,this.overrideClearColor=null,this.overrideClearAlpha=-1}setClearFlags(t,e,n){this.color=t,this.depth=e,this.stencil=n}getOverrideClearColor(){return this.overrideClearColor}setOverrideClearColor(t){this.overrideClearColor=t}getOverrideClearAlpha(){return this.overrideClearAlpha}setOverrideClearAlpha(t){this.overrideClearAlpha=t}render(t,e,n,i,s){let r=this.overrideClearColor,a=this.overrideClearAlpha,o=t.getClearAlpha(),l=r!==null,c=a>=0;l?(t.getClearColor(LA),t.setClearColor(r,c?a:o)):c&&t.setClearAlpha(a),t.setRenderTarget(this.renderToScreen?null:e),t.clear(this.color,this.depth,this.stencil),l?t.setClearColor(LA,o):c&&t.setClearAlpha(o)}},bR=class extends Mr{constructor(t,e){super("MaskPass",t,e),this.needsSwap=!1,this.clearPass=new FA(!1,!1,!0),this.inverse=!1}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get inverted(){return this.inverse}set inverted(t){this.inverse=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getClearPass(){return this.clearPass}isInverted(){return this.inverted}setInverted(t){this.inverted=t}render(t,e,n,i,s){let r=t.getContext(),a=t.state.buffers,o=this.scene,l=this.camera,c=this.clearPass,h=this.inverted?0:1,p=1-h;a.color.setMask(!1),a.depth.setMask(!1),a.color.setLocked(!0),a.depth.setLocked(!0),a.stencil.setTest(!0),a.stencil.setOp(r.REPLACE,r.REPLACE,r.REPLACE),a.stencil.setFunc(r.ALWAYS,h,4294967295),a.stencil.setClear(p),a.stencil.setLocked(!0),this.clearPass.enabled&&(this.renderToScreen?c.render(t,null):(c.render(t,e),c.render(t,n))),this.renderToScreen?(t.setRenderTarget(null),t.render(o,l)):(t.setRenderTarget(e),t.render(o,l),t.setRenderTarget(n),t.render(o,l)),a.color.setLocked(!1),a.depth.setLocked(!1),a.stencil.setLocked(!1),a.stencil.setFunc(r.EQUAL,1,4294967295),a.stencil.setOp(r.KEEP,r.KEEP,r.KEEP),a.stencil.setLocked(!0)}},Jg=1/1e3,wR=1e3,CR=class{constructor(){this.startTime=performance.now(),this.previousTime=0,this.currentTime=0,this._delta=0,this._elapsed=0,this._fixedDelta=1e3/60,this.timescale=1,this.useFixedDelta=!1,this._autoReset=!1}get autoReset(){return this._autoReset}set autoReset(t){typeof document<"u"&&document.hidden!==void 0&&(t?document.addEventListener("visibilitychange",this):document.removeEventListener("visibilitychange",this),this._autoReset=t)}get delta(){return this._delta*Jg}get fixedDelta(){return this._fixedDelta*Jg}set fixedDelta(t){this._fixedDelta=t*wR}get elapsed(){return this._elapsed*Jg}update(t){this.useFixedDelta?this._delta=this.fixedDelta:(this.previousTime=this.currentTime,this.currentTime=(t!==void 0?t:performance.now())-this.startTime,this._delta=this.currentTime-this.previousTime),this._delta*=this.timescale,this._elapsed+=this._delta}reset(){this._delta=0,this._elapsed=0,this.currentTime=performance.now()-this.startTime}getDelta(){return this.delta}getElapsed(){return this.elapsed}handleEvent(t){document.hidden||(this.currentTime=performance.now()-this.startTime)}dispose(){this.autoReset=!1}},zA=class{constructor(t=null,{depthBuffer:e=!0,stencilBuffer:n=!1,multisampling:i=0,frameBufferType:s}={}){this.renderer=null,this.inputBuffer=this.createBuffer(e,n,s,i),this.outputBuffer=this.inputBuffer.clone(),this.copyPass=new TR,this.depthRenderTarget=null,this.passes=[],this.timer=new CR,this.autoRenderToScreen=!0,this.setRenderer(t)}get stableDepthTexture(){return this.depthRenderTarget===null?null:this.depthRenderTarget.depthTexture}get multisampling(){return this.inputBuffer.samples}set multisampling(t){this.multisampling!==t&&(this.inputBuffer.samples=t,this.outputBuffer.samples=t,this.inputBuffer.dispose(),this.outputBuffer.dispose())}getTimer(){return this.timer}getRenderer(){return this.renderer}setRenderer(t){if(this.renderer=t,t!==null){let e=t.getSize(new Ie),n=t.getContext().getContextAttributes().alpha,i=this.inputBuffer.texture.type;i===jt&&t.outputColorSpace===Tt&&(this.inputBuffer.texture.colorSpace=Tt,this.outputBuffer.texture.colorSpace=Tt,this.inputBuffer.dispose(),this.outputBuffer.dispose()),t.autoClear=!1,this.setSize(e.width,e.height);for(let s of this.passes)s.initialize(t,n,i)}}replaceRenderer(t,e=!0){let n=this.renderer,i=n.domElement.parentNode;return this.setRenderer(t),e&&i!==null&&(i.removeChild(n.domElement),i.appendChild(t.domElement)),n}createDepthTexture(){let t=new Ri;t.name="EffectComposer.InputDepth",this.inputBuffer.stencilBuffer?(t.format=Zi,t.type=_r):t.type=ti;let e=t.clone();e.name="EffectComposer.OutputDepth";let n=t.clone();n.name="EffectComposer.StableDepth",this.inputBuffer.depthTexture=t,this.outputBuffer.depthTexture=e,this.inputBuffer.dispose(),this.outputBuffer.dispose();let{width:i,height:s}=this.inputBuffer;this.depthRenderTarget=new Nt(i,s,{depthBuffer:!0,stencilBuffer:this.inputBuffer.stencilBuffer,depthTexture:n})}blitDepthBuffer(t){let e=this.renderer,n=this.depthRenderTarget,i=e.properties,s=e.getContext();e.setRenderTarget(n);let r=i.get(t).__webglFramebuffer,a=i.get(n).__webglFramebuffer,o=t.stencilBuffer?s.DEPTH_BUFFER_BIT|s.STENCIL_BUFFER_BIT:s.DEPTH_BUFFER_BIT;s.bindFramebuffer(s.READ_FRAMEBUFFER,r),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,a),s.blitFramebuffer(0,0,t.width,t.height,0,0,n.width,n.height,o,s.NEAREST),s.bindFramebuffer(s.READ_FRAMEBUFFER,null),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,null),e.setRenderTarget(null)}deleteDepthTexture(){let t=this.stableDepthTexture;for(let e of this.passes)e.getDepthTexture()===t&&e.setDepthTexture(null);this.depthRenderTarget!==null&&(this.depthRenderTarget.dispose(),this.depthRenderTarget=null),this.inputBuffer.depthTexture!==null&&(this.inputBuffer.depthTexture.dispose(),this.inputBuffer.depthTexture=null),this.outputBuffer.depthTexture!==null&&(this.outputBuffer.depthTexture.dispose(),this.outputBuffer.depthTexture=null)}createBuffer(t,e,n,i){let s=this.renderer,r=s===null?new Ie:s.getDrawingBufferSize(new Ie),a=new Nt(r.width,r.height,{minFilter:xt,magFilter:xt,samples:i,stencilBuffer:e,depthBuffer:t,type:n});return n===jt&&s!==null&&s.outputColorSpace===Tt&&(a.texture.colorSpace=Tt),a.texture.name="EffectComposer.Buffer",a.texture.generateMipmaps=!1,a}setMainScene(t){for(let e of this.passes)e.mainScene=t}setMainCamera(t){for(let e of this.passes)e.mainCamera=t}addPass(t,e){let n=this.passes,i=this.renderer,s=i.getDrawingBufferSize(new Ie),r=i.getContext().getContextAttributes().alpha,a=this.inputBuffer.texture.type;if(t.renderer=i,t.setSize(s.width,s.height),t.initialize(i,r,a),this.autoRenderToScreen&&(n.length>0&&(n[n.length-1].renderToScreen=!1),t.renderToScreen&&(this.autoRenderToScreen=!1)),e!==void 0?n.splice(e,0,t):n.push(t),this.autoRenderToScreen&&(n[n.length-1].renderToScreen=!0),t.needsDepthTexture||this.depthRenderTarget!==null)if(this.depthRenderTarget===null){this.createDepthTexture();for(let o of n)o.setDepthTexture(this.stableDepthTexture)}else t.setDepthTexture(this.stableDepthTexture)}removePass(t){let e=this.passes,n=e.indexOf(t);if(n!==-1&&e.splice(n,1).length>0){let r=this.stableDepthTexture;if(r!==null){let a=(l,c)=>l||c.needsDepthTexture;e.reduce(a,!1)||(t.getDepthTexture()===r&&t.setDepthTexture(null),this.deleteDepthTexture())}this.autoRenderToScreen&&n===e.length&&(t.renderToScreen=!1,e.length>0&&(e[e.length-1].renderToScreen=!0))}}removeAllPasses(){let t=this.passes;this.deleteDepthTexture(),t.length>0&&(this.autoRenderToScreen&&(t[t.length-1].renderToScreen=!1),this.passes=[])}render(t){let e=this.renderer,n=this.copyPass,i=this.inputBuffer,s=this.outputBuffer,r,a=!1;t===void 0&&(this.timer.update(),t=this.timer.getDelta());for(let o of this.passes)if(o.enabled){if(o.render(e,i,s,t,a),o.needsDepthBlit&&this.depthRenderTarget!==null&&this.blitDepthBuffer(i),o.needsSwap){if(a){n.renderToScreen=o.renderToScreen;let l=e.getContext(),c=e.state.buffers.stencil;c.setFunc(l.NOTEQUAL,1,4294967295),n.render(e,i,s,t,a),c.setFunc(l.EQUAL,1,4294967295)}r=i,i=s,s=r}o instanceof bR?a=!0:o instanceof SR&&(a=!1)}}setSize(t,e,n){let i=this.renderer,s=i.getSize(new Ie);(t===void 0||e===void 0)&&(t=s.width,e=s.height),(s.width!==t||s.height!==e)&&i.setSize(t,e,n);let r=i.getDrawingBufferSize(new Ie);this.inputBuffer.setSize(r.width,r.height),this.outputBuffer.setSize(r.width,r.height),this.depthRenderTarget!==null&&this.depthRenderTarget.setSize(r.width,r.height);for(let a of this.passes)a.setSize(r.width,r.height)}reset(){this.dispose(),this.autoRenderToScreen=!0}dispose(){for(let t of this.passes)t.dispose();this.deleteDepthTexture(),this.inputBuffer.dispose(),this.outputBuffer.dispose(),this.copyPass.dispose(),this.timer.dispose(),this.passes=[],Mr.fullscreenGeometry.dispose()}},aa={NONE:0,DEPTH:1,CONVOLUTION:2},et={FRAGMENT_HEAD:"FRAGMENT_HEAD",FRAGMENT_MAIN_UV:"FRAGMENT_MAIN_UV",FRAGMENT_MAIN_IMAGE:"FRAGMENT_MAIN_IMAGE",VERTEX_HEAD:"VERTEX_HEAD",VERTEX_MAIN_SUPPORT:"VERTEX_MAIN_SUPPORT"},RR=class{constructor(){this.shaderParts=new Map([[et.FRAGMENT_HEAD,null],[et.FRAGMENT_MAIN_UV,null],[et.FRAGMENT_MAIN_IMAGE,null],[et.VERTEX_HEAD,null],[et.VERTEX_MAIN_SUPPORT,null]]),this.defines=new Map,this.uniforms=new Map,this.blendModes=new Map,this.extensions=new Set,this.attributes=aa.NONE,this.varyings=new Set,this.uvTransformation=!1,this.readDepth=!1,this.colorSpace=Ts}};var jg=!1,NA=class{constructor(t=null){this.originalMaterials=new Map,this.material=null,this.materials=null,this.materialsBackSide=null,this.materialsDoubleSide=null,this.materialsFlatShaded=null,this.materialsFlatShadedBackSide=null,this.materialsFlatShadedDoubleSide=null,this.setMaterial(t),this.meshCount=0,this.replaceMaterial=e=>{if(e.isMesh){let n;if(e.material.flatShading)switch(e.material.side){case Rn:n=this.materialsFlatShadedDoubleSide;break;case Jt:n=this.materialsFlatShadedBackSide;break;default:n=this.materialsFlatShaded;break}else switch(e.material.side){case Rn:n=this.materialsDoubleSide;break;case Jt:n=this.materialsBackSide;break;default:n=this.materials;break}this.originalMaterials.set(e,e.material),e.isSkinnedMesh?e.material=n[2]:e.isInstancedMesh?e.material=n[1]:e.material=n[0],++this.meshCount}}}cloneMaterial(t){if(!(t instanceof Vt))return t.clone();let e=t.uniforms,n=new Map;for(let s in e){let r=e[s].value;r.isRenderTargetTexture&&(e[s].value=null,n.set(s,r))}let i=t.clone();for(let s of n)e[s[0]].value=s[1],i.uniforms[s[0]].value=s[1];return i}setMaterial(t){if(this.disposeMaterials(),this.material=t,t!==null){let e=this.materials=[this.cloneMaterial(t),this.cloneMaterial(t),this.cloneMaterial(t)];for(let n of e)n.uniforms=Object.assign({},t.uniforms),n.side=wi;e[2].skinning=!0,this.materialsBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=Jt,i}),this.materialsDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=Rn,i}),this.materialsFlatShaded=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i}),this.materialsFlatShadedBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=Jt,i}),this.materialsFlatShadedDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=Rn,i})}}render(t,e,n){let i=t.shadowMap.enabled;if(t.shadowMap.enabled=!1,jg){let s=this.originalMaterials;this.meshCount=0,e.traverse(this.replaceMaterial),t.render(e,n);for(let r of s)r[0].material=r[1];this.meshCount!==s.size&&s.clear()}else{let s=e.overrideMaterial;e.overrideMaterial=this.material,t.render(e,n),e.overrideMaterial=s}t.shadowMap.enabled=i}disposeMaterials(){if(this.material!==null){let t=this.materials.concat(this.materialsBackSide).concat(this.materialsDoubleSide).concat(this.materialsFlatShaded).concat(this.materialsFlatShadedBackSide).concat(this.materialsFlatShadedDoubleSide);for(let e of t)e.dispose()}}dispose(){this.originalMaterials.clear(),this.disposeMaterials()}static get workaroundEnabled(){return jg}static set workaroundEnabled(t){jg=t}};var Ze={SKIP:9,SET:30,ADD:0,ALPHA:1,AVERAGE:2,COLOR:3,COLOR_BURN:4,COLOR_DODGE:5,DARKEN:6,DIFFERENCE:7,DIVIDE:8,DST:9,EXCLUSION:10,HARD_LIGHT:11,HARD_MIX:12,HUE:13,INVERT:14,INVERT_RGB:15,LIGHTEN:16,LINEAR_BURN:17,LINEAR_DODGE:18,LINEAR_LIGHT:19,LUMINOSITY:20,MULTIPLY:21,NEGATION:22,NORMAL:23,OVERLAY:24,PIN_LIGHT:25,REFLECT:26,SATURATION:27,SCREEN:28,SOFT_LIGHT:29,SRC:30,SUBTRACT:31,VIVID_LIGHT:32},DR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",UR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,src.a*opacity);}",BR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=(dst.rgb+src.rgb)*0.5;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",IR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.xy,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",PR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=mix(step(0.0,b)*(1.0-min(vec3(1.0),(1.0-a)/max(b,1e-9))),vec3(1.0),step(1.0,a));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",LR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=step(0.0,a)*mix(min(vec3(1.0),a/max(1.0-b,1e-9)),vec3(1.0),step(1.0,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",NR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",OR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=abs(dst.rgb-src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",FR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb/max(src.rgb,1e-9);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",zR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-2.0*dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",GR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb,1.0);vec3 b=min(src.rgb,1.0);vec3 c=mix(2.0*a*b,1.0-2.0*(1.0-a)*(1.0-b),step(0.5,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",HR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=step(1.0,dst.rgb+src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",VR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.x,a.yz));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",kR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",WR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=src.rgb*max(1.0-dst.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",XR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",YR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",qR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb+src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",QR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(2.0*src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",ZR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.xy,b.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",KR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",JR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-abs(1.0-dst.rgb-src.rgb),0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",jR="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,opacity);}",$R="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=2.0*src.rgb*dst.rgb;vec3 b=1.0-2.0*(1.0-src.rgb)*(1.0-dst.rgb);vec3 c=mix(a,b,step(0.5,dst.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",eD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 c=mix(mix(src2,dst.rgb,step(0.5*dst.rgb,src.rgb)),max(src2-1.0,vec3(0.0)),step(dst.rgb,src2-1.0));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",tD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb*dst.rgb/max(1.0-src.rgb,1e-9),1.0);vec3 c=mix(a,src.rgb,step(1.0,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",nD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.x,b.y,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",iD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-min(dst.rgb*src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",sD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 d=dst.rgb+(src2-1.0);vec3 w=step(0.5,src.rgb);vec3 a=dst.rgb-(1.0-src2)*dst.rgb*(1.0-dst.rgb);vec3 b=mix(d*(sqrt(dst.rgb)-dst.rgb),d*dst.rgb*((16.0*dst.rgb-12.0)*dst.rgb+3.0),w*(1.0-step(0.25,dst.rgb)));vec3 c=mix(a,b,w);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",rD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return src;}",aD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",oD="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=mix(max(1.0-min((1.0-dst.rgb)/(2.0*src.rgb),1.0),0.0),min(dst.rgb/(2.0*(1.0-src.rgb)),1.0),step(0.5,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",lD=new Map([[Ze.ADD,DR],[Ze.ALPHA,UR],[Ze.AVERAGE,BR],[Ze.COLOR,IR],[Ze.COLOR_BURN,PR],[Ze.COLOR_DODGE,LR],[Ze.DARKEN,NR],[Ze.DIFFERENCE,OR],[Ze.DIVIDE,FR],[Ze.DST,null],[Ze.EXCLUSION,zR],[Ze.HARD_LIGHT,GR],[Ze.HARD_MIX,HR],[Ze.HUE,VR],[Ze.INVERT,kR],[Ze.INVERT_RGB,WR],[Ze.LIGHTEN,XR],[Ze.LINEAR_BURN,YR],[Ze.LINEAR_DODGE,qR],[Ze.LINEAR_LIGHT,QR],[Ze.LUMINOSITY,ZR],[Ze.MULTIPLY,KR],[Ze.NEGATION,JR],[Ze.NORMAL,jR],[Ze.OVERLAY,$R],[Ze.PIN_LIGHT,eD],[Ze.REFLECT,tD],[Ze.SATURATION,nD],[Ze.SCREEN,iD],[Ze.SOFT_LIGHT,sD],[Ze.SRC,rD],[Ze.SUBTRACT,aD],[Ze.VIVID_LIGHT,oD]]),cD=class extends zn{constructor(t,e=1){super(),this._blendFunction=t,this.opacity=new Ct(e)}getOpacity(){return this.opacity.value}setOpacity(t){this.opacity.value=t}get blendFunction(){return this._blendFunction}set blendFunction(t){this._blendFunction=t,this.dispatchEvent({type:"change"})}getBlendFunction(){return this.blendFunction}setBlendFunction(t){this.blendFunction=t}getShaderCode(){return lD.get(this.blendFunction)}};var GA=class extends zn{constructor(t,e,{attributes:n=aa.NONE,blendFunction:i=Ze.NORMAL,defines:s=new Map,uniforms:r=new Map,extensions:a=null,vertexShader:o=null}={}){super(),this.name=t,this.renderer=null,this.attributes=n,this.fragmentShader=e,this.vertexShader=o,this.defines=s,this.uniforms=r,this.extensions=a,this.blendMode=new cD(i),this.blendMode.addEventListener("change",l=>this.setChanged()),this._inputColorSpace=Ts,this._outputColorSpace=ni}get inputColorSpace(){return this._inputColorSpace}set inputColorSpace(t){this._inputColorSpace=t,this.setChanged()}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t,this.setChanged()}set mainScene(t){}set mainCamera(t){}getName(){return this.name}setRenderer(t){this.renderer=t}getDefines(){return this.defines}getUniforms(){return this.uniforms}getExtensions(){return this.extensions}getBlendMode(){return this.blendMode}getAttributes(){return this.attributes}setAttributes(t){this.attributes=t,this.setChanged()}getFragmentShader(){return this.fragmentShader}setFragmentShader(t){this.fragmentShader=t,this.setChanged()}getVertexShader(){return this.vertexShader}setVertexShader(t){this.vertexShader=t,this.setChanged()}setChanged(){this.dispatchEvent({type:"change"})}setDepthTexture(t,e=Ki){}update(t,e,n){}setSize(t,e){}initialize(t,e,n){}dispose(){for(let t of Object.keys(this)){let e=this[t];(e instanceof Nt||e instanceof Ci||e instanceof Kt||e instanceof Mr)&&this[t].dispose()}}};var bI=[new Float32Array([0,0]),new Float32Array([0,1,1]),new Float32Array([0,1,1,2]),new Float32Array([0,1,2,2,3]),new Float32Array([0,1,2,3,4,4,5]),new Float32Array([0,1,2,3,4,5,7,8,9,10])];var HA=class extends Mr{constructor(t,e,n=null){super("RenderPass",t,e),this.needsSwap=!1,this.needsDepthBlit=!0,this.clearPass=new FA,this.overrideMaterialManager=n===null?null:new NA(n),this.ignoreBackground=!1,this.skipShadowMapUpdate=!1,this.selection=null}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get renderToScreen(){return super.renderToScreen}set renderToScreen(t){super.renderToScreen=t,this.clearPass.renderToScreen=t}get overrideMaterial(){let t=this.overrideMaterialManager;return t!==null?t.material:null}set overrideMaterial(t){let e=this.overrideMaterialManager;t!==null?e!==null?e.setMaterial(t):this.overrideMaterialManager=new NA(t):e!==null&&(e.dispose(),this.overrideMaterialManager=null)}getOverrideMaterial(){return this.overrideMaterial}setOverrideMaterial(t){this.overrideMaterial=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getSelection(){return this.selection}setSelection(t){this.selection=t}isBackgroundDisabled(){return this.ignoreBackground}setBackgroundDisabled(t){this.ignoreBackground=t}isShadowMapDisabled(){return this.skipShadowMapUpdate}setShadowMapDisabled(t){this.skipShadowMapUpdate=t}getClearPass(){return this.clearPass}render(t,e,n,i,s){let r=this.scene,a=this.camera,o=this.selection,l=a.layers.mask,c=r.background,h=t.shadowMap.autoUpdate,p=this.renderToScreen?null:e;o!==null&&a.layers.set(o.getLayer()),this.skipShadowMapUpdate&&(t.shadowMap.autoUpdate=!1),(this.ignoreBackground||this.clearPass.overrideClearColor!==null)&&(r.background=null),this.clearPass.enabled&&this.clearPass.render(t,e),t.setRenderTarget(p),this.overrideMaterialManager!==null?this.overrideMaterialManager.render(t,r,a):t.render(r,a),a.layers.mask=l,r.background=c,t.shadowMap.autoUpdate=h}};var wI=Math.PI*.5;var uD=`#include <common>
#include <packing>
#include <dithering_pars_fragment>
#define packFloatToRGBA(v) packDepthToRGBA(v)
#define unpackRGBAToFloat(v) unpackRGBAToDepth(v)
#ifdef FRAMEBUFFER_PRECISION_HIGH
uniform mediump sampler2D inputBuffer;
#else
uniform lowp sampler2D inputBuffer;
#endif
#if DEPTH_PACKING == 3201
uniform lowp sampler2D depthBuffer;
#elif defined(GL_FRAGMENT_PRECISION_HIGH)
uniform highp sampler2D depthBuffer;
#else
uniform mediump sampler2D depthBuffer;
#endif
uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;vec4 sRGBToLinear(const in vec4 value){return vec4(mix(pow(value.rgb*0.9478672986+vec3(0.0521327014),vec3(2.4)),value.rgb*0.0773993808,vec3(lessThanEqual(value.rgb,vec3(0.04045)))),value.a);}float readDepth(const in vec2 uv){
#if DEPTH_PACKING == 3201
float depth=unpackRGBAToDepth(texture2D(depthBuffer,uv));
#else
float depth=texture2D(depthBuffer,uv).r;
#endif
#if defined(USE_LOGARITHMIC_DEPTH_BUFFER) || defined(LOG_DEPTH)
float d=pow(2.0,depth*log2(cameraFar+1.0))-1.0;float a=cameraFar/(cameraFar-cameraNear);float b=cameraFar*cameraNear/(cameraNear-cameraFar);depth=a+b/d;
#elif defined(USE_REVERSED_DEPTH_BUFFER)
depth=1.0-depth;
#endif
return depth;}float getViewZ(const in float depth){
#ifdef PERSPECTIVE_CAMERA
return perspectiveDepthToViewZ(depth,cameraNear,cameraFar);
#else
return orthographicDepthToViewZ(depth,cameraNear,cameraFar);
#endif
}vec3 RGBToHCV(const in vec3 RGB){vec4 P=mix(vec4(RGB.bg,-1.0,2.0/3.0),vec4(RGB.gb,0.0,-1.0/3.0),step(RGB.b,RGB.g));vec4 Q=mix(vec4(P.xyw,RGB.r),vec4(RGB.r,P.yzx),step(P.x,RGB.r));float C=Q.x-min(Q.w,Q.y);float H=abs((Q.w-Q.y)/(6.0*C+EPSILON)+Q.z);return vec3(H,C,Q.x);}vec3 RGBToHSL(const in vec3 RGB){vec3 HCV=RGBToHCV(RGB);float L=HCV.z-HCV.y*0.5;float S=HCV.y/(1.0-abs(L*2.0-1.0)+EPSILON);return vec3(HCV.x,S,L);}vec3 HueToRGB(const in float H){float R=abs(H*6.0-3.0)-1.0;float G=2.0-abs(H*6.0-2.0);float B=2.0-abs(H*6.0-4.0);return clamp(vec3(R,G,B),0.0,1.0);}vec3 HSLToRGB(const in vec3 HSL){vec3 RGB=HueToRGB(HSL.x);float C=(1.0-abs(2.0*HSL.z-1.0))*HSL.y;return(RGB-0.5)*C+HSL.z;}FRAGMENT_HEAD void main(){FRAGMENT_MAIN_UV vec4 color0=texture2D(inputBuffer,UV);vec4 color1=vec4(0.0);FRAGMENT_MAIN_IMAGE color0.a=clamp(color0.a,0.0,1.0);gl_FragColor=color0;
#ifdef ENCODE_OUTPUT
#include <colorspace_fragment>
#endif
#include <dithering_fragment>
}`,fD="uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;VERTEX_HEAD void main(){vUv=position.xy*0.5+0.5;VERTEX_MAIN_SUPPORT gl_Position=vec4(position.xy,1.0,1.0);}",hD=class extends Vt{constructor(t,e,n,i,s=!1){super({name:"EffectMaterial",defines:{THREE_REVISION:"185".replace(/\D+/g,""),DEPTH_PACKING:"0",ENCODE_OUTPUT:"1"},uniforms:{inputBuffer:new Ct(null),depthBuffer:new Ct(null),resolution:new Ct(new Ie),texelSize:new Ct(new Ie),cameraNear:new Ct(.3),cameraFar:new Ct(1e3),aspect:new Ct(1),time:new Ct(0)},blending:Gn,toneMapped:!1,depthWrite:!1,depthTest:!1,dithering:s}),t&&this.setShaderParts(t),e&&this.setDefines(e),n&&this.setUniforms(n),this.copyCameraSettings(i)}set inputBuffer(t){this.uniforms.inputBuffer.value=t}setInputBuffer(t){this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){this.uniforms.depthBuffer.value=t}get depthPacking(){return Number(this.defines.DEPTH_PACKING)}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}setDepthBuffer(t,e=Ki){this.depthBuffer=t,this.depthPacking=e}setShaderData(t){this.setShaderParts(t.shaderParts),this.setDefines(t.defines),this.setUniforms(t.uniforms),this.setExtensions(t.extensions)}setShaderParts(t){return this.fragmentShader=uD.replace(et.FRAGMENT_HEAD,t.get(et.FRAGMENT_HEAD)||"").replace(et.FRAGMENT_MAIN_UV,t.get(et.FRAGMENT_MAIN_UV)||"").replace(et.FRAGMENT_MAIN_IMAGE,t.get(et.FRAGMENT_MAIN_IMAGE)||""),this.vertexShader=fD.replace(et.VERTEX_HEAD,t.get(et.VERTEX_HEAD)||"").replace(et.VERTEX_MAIN_SUPPORT,t.get(et.VERTEX_MAIN_SUPPORT)||""),this.needsUpdate=!0,this}setDefines(t){for(let e of t.entries())this.defines[e[0]]=e[1];return this.needsUpdate=!0,this}setUniforms(t){for(let e of t.entries())this.uniforms[e[0]]=e[1];return this}setExtensions(t){this.extensions={};for(let e of t)this.extensions[e]=!0;return this}get encodeOutput(){return this.defines.ENCODE_OUTPUT!==void 0}set encodeOutput(t){this.encodeOutput!==t&&(t?this.defines.ENCODE_OUTPUT="1":delete this.defines.ENCODE_OUTPUT,this.needsUpdate=!0)}isOutputEncodingEnabled(t){return this.encodeOutput}setOutputEncodingEnabled(t){this.encodeOutput=t}get time(){return this.uniforms.time.value}set time(t){this.uniforms.time.value=t}setDeltaTime(t){this.uniforms.time.value+=t}adoptCameraSettings(t){this.copyCameraSettings(t)}copyCameraSettings(t){t&&(this.uniforms.cameraNear.value=t.near,this.uniforms.cameraFar.value=t.far,t instanceof Sn?this.defines.PERSPECTIVE_CAMERA="1":delete this.defines.PERSPECTIVE_CAMERA,this.needsUpdate=!0)}setSize(t,e){let n=this.uniforms;n.resolution.value.set(t,e),n.texelSize.value.set(1/t,1/e),n.aspect.value=t/e}static get Section(){return et}};var RI=Number("185".replace(/\D+/g,"")),ra=255/256,DI=new Float32Array([ra/256**3,ra/256**2,ra/256,ra]),UI=new Float32Array([ra,ra/256,ra/256**2,1/256**3]);function OA(t,e,n){for(let i of e){let s="$1"+t+i.charAt(0).toUpperCase()+i.slice(1),r=new RegExp("([^\\.])(\\b"+i+"\\b)","g");for(let a of n.entries())a[1]!==null&&n.set(a[0],a[1].replace(r,s))}}function dD(t,e,n){let i=e.getFragmentShader(),s=e.getVertexShader(),r=i!==void 0&&/mainImage/.test(i),a=i!==void 0&&/mainUv/.test(i);if(n.attributes|=e.getAttributes(),i===void 0)throw new Error(`Missing fragment shader (${e.name})`);if(a&&(n.attributes&aa.CONVOLUTION)!==0)throw new Error(`Effects that transform UVs are incompatible with convolution effects (${e.name})`);if(!r&&!a)throw new Error(`Could not find mainImage or mainUv function (${e.name})`);{let o=/\w+\s+(\w+)\([\w\s,]*\)\s*{/g,l=n.shaderParts,c=l.get(et.FRAGMENT_HEAD)||"",h=l.get(et.FRAGMENT_MAIN_UV)||"",p=l.get(et.FRAGMENT_MAIN_IMAGE)||"",u=l.get(et.VERTEX_HEAD)||"",d=l.get(et.VERTEX_MAIN_SUPPORT)||"",v=new Set,M=new Set;if(a&&(h+=`	${t}MainUv(UV);
`,n.uvTransformation=!0),s!==null&&/mainSupport/.test(s)){let g=/mainSupport *\([\w\s]*?uv\s*?\)/.test(s);d+=`	${t}MainSupport(`,d+=g?`vUv);
`:`);
`;for(let S of s.matchAll(/(?:varying\s+\w+\s+([\S\s]*?);)/g))for(let _ of S[1].split(/\s*,\s*/))n.varyings.add(_),v.add(_),M.add(_);for(let S of s.matchAll(o))M.add(S[1])}for(let g of i.matchAll(o))M.add(g[1]);for(let g of e.defines.keys())M.add(g.replace(/\([\w\s,]*\)/g,""));for(let g of e.uniforms.keys())M.add(g);M.delete("while"),M.delete("for"),M.delete("if"),e.uniforms.forEach((g,S)=>n.uniforms.set(t+S.charAt(0).toUpperCase()+S.slice(1),g)),e.defines.forEach((g,S)=>n.defines.set(t+S.charAt(0).toUpperCase()+S.slice(1),g));let m=new Map([["fragment",i],["vertex",s]]);OA(t,M,n.defines),OA(t,M,m),i=m.get("fragment"),s=m.get("vertex");let f=e.blendMode;if(n.blendModes.set(f.blendFunction,f),r){e.inputColorSpace!==null&&e.inputColorSpace!==n.colorSpace&&(p+=e.inputColorSpace===Tt?`color0 = sRGBTransferOETF(color0);
	`:`color0 = sRGBToLinear(color0);
	`),e.outputColorSpace!==ni?n.colorSpace=e.outputColorSpace:e.inputColorSpace!==null&&(n.colorSpace=e.inputColorSpace);let g=/MainImage *\([\w\s,]*?depth[\w\s,]*?\)/;p+=`${t}MainImage(color0, UV, `,(n.attributes&aa.DEPTH)!==0&&g.test(i)&&(p+="depth, ",n.readDepth=!0),p+=`color1);
	`;let S=t+"BlendOpacity";n.uniforms.set(S,f.opacity),p+=`color0 = blend${f.blendFunction}(color0, color1, ${S});

	`,c+=`uniform float ${S};

`}if(c+=i+`
`,s!==null&&(u+=s+`
`),l.set(et.FRAGMENT_HEAD,c),l.set(et.FRAGMENT_MAIN_UV,h),l.set(et.FRAGMENT_MAIN_IMAGE,p),l.set(et.VERTEX_HEAD,u),l.set(et.VERTEX_MAIN_SUPPORT,d),e.extensions!==null)for(let g of e.extensions)n.extensions.add(g)}}var VA=class extends Mr{constructor(t,...e){super("EffectPass"),this.fullscreenMaterial=new hD(null,null,null,t),this.listener=n=>this.handleEvent(n),this.effects=[],this.setEffects(e),this.skipRendering=!1,this.minTime=1,this.maxTime=Number.POSITIVE_INFINITY,this.timeScale=1}set mainScene(t){for(let e of this.effects)e.mainScene=t}set mainCamera(t){this.fullscreenMaterial.copyCameraSettings(t);for(let e of this.effects)e.mainCamera=t}get encodeOutput(){return this.fullscreenMaterial.encodeOutput}set encodeOutput(t){this.fullscreenMaterial.encodeOutput=t}get dithering(){return this.fullscreenMaterial.dithering}set dithering(t){let e=this.fullscreenMaterial;e.dithering=t,e.needsUpdate=!0}setEffects(t){for(let e of this.effects)e.removeEventListener("change",this.listener);this.effects=t.sort((e,n)=>n.attributes-e.attributes);for(let e of this.effects)e.addEventListener("change",this.listener)}updateMaterial(){let t=new RR,e=0;for(let a of this.effects)if(a.blendMode.blendFunction===Ze.DST)t.attributes|=a.getAttributes()&aa.DEPTH;else{if((t.attributes&a.getAttributes()&aa.CONVOLUTION)!==0)throw new Error(`Convolution effects cannot be merged (${a.name})`);dD("e"+e++,a,t)}let n=t.shaderParts.get(et.FRAGMENT_HEAD),i=t.shaderParts.get(et.FRAGMENT_MAIN_IMAGE),s=t.shaderParts.get(et.FRAGMENT_MAIN_UV),r=/\bblend\b/g;for(let a of t.blendModes.values())n+=a.getShaderCode().replace(r,`blend${a.blendFunction}`)+`
`;(t.attributes&aa.DEPTH)!==0?(t.readDepth&&(i=`float depth = readDepth(UV);

	`+i),this.needsDepthTexture=this.getDepthTexture()===null):this.needsDepthTexture=!1,t.colorSpace===Tt&&(i+=`color0 = sRGBToLinear(color0);
	`),t.uvTransformation?(s=`vec2 transformedUv = vUv;
`+s,t.defines.set("UV","transformedUv")):t.defines.set("UV","vUv"),t.shaderParts.set(et.FRAGMENT_HEAD,n),t.shaderParts.set(et.FRAGMENT_MAIN_IMAGE,i),t.shaderParts.set(et.FRAGMENT_MAIN_UV,s);for(let[a,o]of t.shaderParts)o!==null&&t.shaderParts.set(a,o.trim().replace(/^#/,`
#`));this.skipRendering=e===0,this.needsSwap=!this.skipRendering,this.fullscreenMaterial.setShaderData(t)}recompile(){this.updateMaterial()}getDepthTexture(){return this.fullscreenMaterial.depthBuffer}setDepthTexture(t,e=Ki){this.fullscreenMaterial.depthBuffer=t,this.fullscreenMaterial.depthPacking=e;for(let n of this.effects)n.setDepthTexture(t,e)}render(t,e,n,i,s){for(let r of this.effects)r.update(t,e,i);if(!this.skipRendering||this.renderToScreen){let r=this.fullscreenMaterial;r.inputBuffer=e.texture,r.time+=i*this.timeScale,t.setRenderTarget(this.renderToScreen?null:n),t.render(this.scene,this.camera)}}setSize(t,e){this.fullscreenMaterial.setSize(t,e);for(let n of this.effects)n.setSize(t,e)}initialize(t,e,n){this.renderer=t;for(let i of this.effects)i.initialize(t,e,n);this.updateMaterial(),n!==void 0&&n!==jt&&(this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1")}dispose(){super.dispose();for(let t of this.effects)t.removeEventListener("change",this.listener),t.dispose()}handleEvent(t){t.type==="change"&&this.recompile()}};var II=[new Float32Array(3),new Float32Array(3)],PI=[new Float32Array(3),new Float32Array(3),new Float32Array(3),new Float32Array(3)],LI=[[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([0,1,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([0,1,1]),new Float32Array([1,1,1])]];var NI=[new Float32Array(2),new Float32Array(2)];var OI=new Float32Array([0,-.25,.25,-.125,.125,-.375,.375]),FI=[new Float32Array([0,0]),new Float32Array([.25,-.25]),new Float32Array([-.25,.25]),new Float32Array([.125,-.125]),new Float32Array([-.125,.125])],zI=[new Uint8Array([0,0]),new Uint8Array([3,0]),new Uint8Array([0,3]),new Uint8Array([3,3]),new Uint8Array([1,0]),new Uint8Array([4,0]),new Uint8Array([1,3]),new Uint8Array([4,3]),new Uint8Array([0,1]),new Uint8Array([3,1]),new Uint8Array([0,4]),new Uint8Array([3,4]),new Uint8Array([1,1]),new Uint8Array([4,1]),new Uint8Array([1,4]),new Uint8Array([4,4])],GI=[new Uint8Array([0,0]),new Uint8Array([1,0]),new Uint8Array([0,2]),new Uint8Array([1,2]),new Uint8Array([2,0]),new Uint8Array([3,0]),new Uint8Array([2,2]),new Uint8Array([3,2]),new Uint8Array([0,1]),new Uint8Array([1,1]),new Uint8Array([0,3]),new Uint8Array([1,3]),new Uint8Array([2,1]),new Uint8Array([3,1]),new Uint8Array([2,3]),new Uint8Array([3,3])];var HI=new Map([[Tn(0,0,0,0),new Float32Array([0,0,0,0])],[Tn(0,0,0,1),new Float32Array([0,0,0,1])],[Tn(0,0,1,0),new Float32Array([0,0,1,0])],[Tn(0,0,1,1),new Float32Array([0,0,1,1])],[Tn(0,1,0,0),new Float32Array([0,1,0,0])],[Tn(0,1,0,1),new Float32Array([0,1,0,1])],[Tn(0,1,1,0),new Float32Array([0,1,1,0])],[Tn(0,1,1,1),new Float32Array([0,1,1,1])],[Tn(1,0,0,0),new Float32Array([1,0,0,0])],[Tn(1,0,0,1),new Float32Array([1,0,0,1])],[Tn(1,0,1,0),new Float32Array([1,0,1,0])],[Tn(1,0,1,1),new Float32Array([1,0,1,1])],[Tn(1,1,0,0),new Float32Array([1,1,0,0])],[Tn(1,1,0,1),new Float32Array([1,1,0,1])],[Tn(1,1,1,0),new Float32Array([1,1,1,0])],[Tn(1,1,1,1),new Float32Array([1,1,1,1])]]);function $g(t,e,n){return t+(e-t)*n}function Tn(t,e,n,i){let s=$g(t,e,.75),r=$g(n,i,1-.25);return $g(s,r,1-.125)}var zh=Tr(bo());var YA=Tr(lc()),gD=()=>{let e=document.createElement("canvas");e.width=64,e.height=64;let n=e.getContext("2d");if(!n)throw new Error("2D context not available");n.fillStyle="black",n.fillRect(0,0,e.width,e.height);let i=new Kt(e);i.minFilter=xt,i.magFilter=xt,i.generateMipmaps=!1;let s=[],r=null,a=64,o=.1*64,l=1/a,c=()=>{n.fillStyle="black",n.fillRect(0,0,e.width,e.height)},h=d=>{let v={x:d.x*64,y:(1-d.y)*64},M=1,m=_=>Math.sin(_*Math.PI/2),f=_=>-_*(_-2);d.age<a*.3?M=m(d.age/(a*.3)):M=f(1-(d.age-a*.3)/(a*.7))||0,M*=d.force;let g=`${(d.vx+1)/2*255}, ${(d.vy+1)/2*255}, ${M*255}`,S=320;n.shadowOffsetX=S,n.shadowOffsetY=S,n.shadowBlur=o,n.shadowColor=`rgba(${g},${.22*M})`,n.beginPath(),n.fillStyle="rgba(255,0,0,1)",n.arc(v.x-S,v.y-S,o,0,Math.PI*2),n.fill()};return{texture:i,addTouch:d=>{let v=0,M=0,m=0;if(r){let f=d.x-r.x,g=d.y-r.y;if(f===0&&g===0)return;let S=f*f+g*g,_=Math.sqrt(S);M=f/(_||1),m=g/(_||1),v=Math.min(S*1e4,1)}r={x:d.x,y:d.y},s.push({x:d.x,y:d.y,age:0,force:v,vx:M,vy:m})},update:()=>{c();for(let d=s.length-1;d>=0;d-=1){let v=s[d],M=v.force*l*(1-v.age/a);v.x+=v.vx*M,v.y+=v.vy*M,v.age+=1,v.age>a&&s.splice(d,1)}s.forEach(h),i.needsUpdate=!0},set radiusScale(d){o=.1*64*d}}},vD=(t,e)=>{let n=`
    uniform sampler2D uTexture;
    uniform float uStrength;
    uniform float uTime;
    uniform float uFreq;

    void mainUv(inout vec2 uv) {
      vec4 tex = texture2D(uTexture, uv);
      float vx = tex.r * 2.0 - 1.0;
      float vy = tex.g * 2.0 - 1.0;
      float intensity = tex.b;
      float wave = 0.5 + 0.5 * sin(uTime * uFreq + intensity * 6.2831853);
      float amount = uStrength * intensity * wave;
      uv += vec2(vx, vy) * amount;
    }
  `;return new GA("LiquidEffect",n,{uniforms:new Map([["uTexture",new Ct(t)],["uStrength",new Ct(e?.strength??.025)],["uTime",new Ct(0)],["uFreq",new Ct(e?.freq??4.5)]])})},xD={square:0,circle:1,triangle:2,diamond:3},yD=`
void main() {
  gl_Position = vec4(position, 1.0);
}
`,_D=`
precision highp float;

uniform vec3  uColor;
uniform vec2  uResolution;
uniform float uTime;
uniform float uPixelSize;
uniform float uScale;
uniform float uDensity;
uniform float uPixelJitter;
uniform int   uEnableRipples;
uniform float uRippleSpeed;
uniform float uRippleThickness;
uniform float uRippleIntensity;
uniform float uEdgeFade;
uniform int   uShapeType;

const int SHAPE_SQUARE   = 0;
const int SHAPE_CIRCLE   = 1;
const int SHAPE_TRIANGLE = 2;
const int SHAPE_DIAMOND  = 3;
const int MAX_CLICKS = 10;

uniform vec2  uClickPos[MAX_CLICKS];
uniform float uClickTimes[MAX_CLICKS];

out vec4 fragColor;

float Bayer2(vec2 a) {
  a = floor(a);
  return fract(a.x / 2.0 + a.y * a.y * 0.75);
}
#define Bayer4(a) (Bayer2(0.5*(a))*0.25 + Bayer2(a))
#define Bayer8(a) (Bayer4(0.5*(a))*0.25 + Bayer2(a))

#define FBM_OCTAVES 5
#define FBM_LACUNARITY 1.25
#define FBM_GAIN 1.0

float hash11(float n) {
  return fract(sin(n) * 43758.5453);
}

float vnoise(vec3 p) {
  vec3 ip = floor(p);
  vec3 fp = fract(p);
  float n000 = hash11(dot(ip + vec3(0.0,0.0,0.0), vec3(1.0,57.0,113.0)));
  float n100 = hash11(dot(ip + vec3(1.0,0.0,0.0), vec3(1.0,57.0,113.0)));
  float n010 = hash11(dot(ip + vec3(0.0,1.0,0.0), vec3(1.0,57.0,113.0)));
  float n110 = hash11(dot(ip + vec3(1.0,1.0,0.0), vec3(1.0,57.0,113.0)));
  float n001 = hash11(dot(ip + vec3(0.0,0.0,1.0), vec3(1.0,57.0,113.0)));
  float n101 = hash11(dot(ip + vec3(1.0,0.0,1.0), vec3(1.0,57.0,113.0)));
  float n011 = hash11(dot(ip + vec3(0.0,1.0,1.0), vec3(1.0,57.0,113.0)));
  float n111 = hash11(dot(ip + vec3(1.0,1.0,1.0), vec3(1.0,57.0,113.0)));
  vec3 w = fp*fp*fp*(fp*(fp*6.0-15.0)+10.0);
  float x00 = mix(n000, n100, w.x);
  float x10 = mix(n010, n110, w.x);
  float x01 = mix(n001, n101, w.x);
  float x11 = mix(n011, n111, w.x);
  float y0 = mix(x00, x10, w.y);
  float y1 = mix(x01, x11, w.y);
  return mix(y0, y1, w.z) * 2.0 - 1.0;
}

float fbm2(vec2 uv, float time) {
  vec3 p = vec3(uv * uScale, time);
  float amplitude = 1.0;
  float frequency = 1.0;
  float sum = 1.0;
  for (int index = 0; index < FBM_OCTAVES; ++index) {
    sum += amplitude * vnoise(p * frequency);
    frequency *= FBM_LACUNARITY;
    amplitude *= FBM_GAIN;
  }
  return sum * 0.5 + 0.5;
}

float maskCircle(vec2 p, float coverage) {
  float radius = sqrt(coverage) * 0.25;
  float distance = length(p - 0.5) - radius;
  float aa = 0.5 * fwidth(distance);
  return coverage * (1.0 - smoothstep(-aa, aa, distance * 2.0));
}

float maskTriangle(vec2 p, vec2 id, float coverage) {
  bool flip = mod(id.x + id.y, 2.0) > 0.5;
  if (flip) p.x = 1.0 - p.x;
  float radius = sqrt(coverage);
  float distance = p.y - radius * (1.0 - p.x);
  float aa = fwidth(distance);
  return coverage * clamp(0.5 - distance / aa, 0.0, 1.0);
}

float maskDiamond(vec2 p, float coverage) {
  float radius = sqrt(coverage) * 0.564;
  return step(abs(p.x - 0.49) + abs(p.y - 0.49), radius);
}

void main() {
  float pixelSize = uPixelSize;
  vec2 fragCoord = gl_FragCoord.xy - uResolution * 0.5;
  float aspectRatio = uResolution.x / uResolution.y;
  vec2 pixelId = floor(fragCoord / pixelSize);
  vec2 pixelUV = fract(fragCoord / pixelSize);
  float cellPixelSize = 8.0 * pixelSize;
  vec2 cellId = floor(fragCoord / cellPixelSize);
  vec2 cellCoord = cellId * cellPixelSize;
  vec2 uv = cellCoord / uResolution * vec2(aspectRatio, 1.0);

  float base = fbm2(uv, uTime * 0.05);
  base = base * 0.5 - 0.65;
  float feed = base + (uDensity - 0.5) * 0.3;

  if (uEnableRipples == 1) {
    for (int index = 0; index < MAX_CLICKS; ++index) {
      vec2 position = uClickPos[index];
      if (position.x < 0.0) continue;
      vec2 clickUv = (
        (position - uResolution * 0.5 - cellPixelSize * 0.5) / uResolution
      ) * vec2(aspectRatio, 1.0);
      float elapsed = max(uTime - uClickTimes[index], 0.0);
      float radius = distance(uv, clickUv);
      float waveRadius = uRippleSpeed * elapsed;
      float ring = exp(-pow((radius - waveRadius) / uRippleThickness, 2.0));
      float attenuation = exp(-elapsed) * exp(-10.0 * radius);
      feed = max(feed, ring * attenuation * uRippleIntensity);
    }
  }

  float bayer = Bayer8(fragCoord / uPixelSize) - 0.5;
  float blackWhite = step(0.5, feed + bayer);
  float hash = fract(
    sin(dot(floor(fragCoord / uPixelSize), vec2(127.1, 311.7))) * 43758.5453
  );
  float jitterScale = 1.0 + (hash - 0.5) * uPixelJitter;
  float coverage = blackWhite * jitterScale;
  float mask;
  if (uShapeType == SHAPE_CIRCLE) {
    mask = maskCircle(pixelUV, coverage);
  } else if (uShapeType == SHAPE_TRIANGLE) {
    mask = maskTriangle(pixelUV, pixelId, coverage);
  } else if (uShapeType == SHAPE_DIAMOND) {
    mask = maskDiamond(pixelUV, coverage);
  } else {
    mask = coverage;
  }

  if (uEdgeFade > 0.0) {
    vec2 normalized = gl_FragCoord.xy / uResolution;
    float edge = min(
      min(normalized.x, normalized.y),
      min(1.0 - normalized.x, 1.0 - normalized.y)
    );
    mask *= smoothstep(0.0, uEdgeFade, edge);
  }

  vec3 srgbColor = mix(
    uColor * 12.92,
    1.055 * pow(uColor, vec3(1.0 / 2.4)) - 0.055,
    step(0.0031308, uColor)
  );
  fragColor = vec4(srgbColor, mask);
}
`,tv=10;function nv({variant:t="square",pixelSize:e=4,color:n="#B497CF",className:i="",style:s,antialias:r=!0,patternScale:a=2,patternDensity:o=1,liquid:l=!1,liquidStrength:c=.1,liquidRadius:h=1,pixelSizeJitter:p=0,enableRipples:u=!0,rippleIntensityScale:d=1,rippleThickness:v=.1,rippleSpeed:M=.3,liquidWobbleSpeed:m=4.5,autoPauseOffscreen:f=!0,speed:g=.5,transparent:S=!0,edgeFade:_=.5}){let T=(0,zh.useRef)(null);return(0,zh.useEffect)(()=>{let b=T.current;if(!b)return;let w=document.createElement("canvas"),x=new Ph({canvas:w,antialias:r,alpha:!0,powerPreference:"high-performance"});x.domElement.style.width="100%",x.domElement.style.height="100%",x.setPixelRatio(Math.min(window.devicePixelRatio||1,2)),b.appendChild(x.domElement),S?x.setClearAlpha(0):x.setClearColor(0,1);let E={uResolution:{value:new Ie(0,0)},uTime:{value:0},uColor:{value:new ke(n)},uClickPos:{value:Array.from({length:tv},()=>new Ie(-1,-1))},uClickTimes:{value:new Float32Array(tv)},uShapeType:{value:xD[t]??0},uPixelSize:{value:e*x.getPixelRatio()},uScale:{value:a},uDensity:{value:o},uPixelJitter:{value:p},uEnableRipples:{value:u?1:0},uRippleSpeed:{value:M},uRippleThickness:{value:v},uRippleIntensity:{value:d},uEdgeFade:{value:_}},R=new dr,D=new bs(-1,1,1,-1,0,1),L=new Vt({vertexShader:yD,fragmentShader:_D,uniforms:E,transparent:!0,depthTest:!1,depthWrite:!1,glslVersion:ic}),q=new Mn(new ta(2,2),L);R.add(q);let Y,N,k;if(l){N=gD(),N.radiusScale=h,Y=new zA(x),Y.addPass(new HA(R,D)),k=vD(N.texture,{strength:c,freq:m});let Se=new VA(D,k);Se.renderToScreen=!0,Y.addPass(Se)}let V=()=>{let Se=b.clientWidth||1,Ae=b.clientHeight||1;x.setSize(Se,Ae,!1),E.uResolution.value.set(x.domElement.width,x.domElement.height),E.uPixelSize.value=e*x.getPixelRatio(),Y?.setSize(x.domElement.width,x.domElement.height)},j=new ResizeObserver(V);j.observe(b),V();let ee=Se=>{let Ae=x.domElement.getBoundingClientRect(),Ce=x.domElement.width/Ae.width,Rt=x.domElement.height/Ae.height;return{x:(Se.clientX-Ae.left)*Ce,y:(Ae.height-(Se.clientY-Ae.top))*Rt,width:x.domElement.width,height:x.domElement.height}},se=0,he=Se=>{let Ae=ee(Se);E.uClickPos.value[se].set(Ae.x,Ae.y),E.uClickTimes.value[se]=E.uTime.value,se=(se+1)%tv},ve=Se=>{if(!N)return;let Ae=ee(Se);N.addTouch({x:Ae.x/Ae.width,y:Ae.y/Ae.height})};window.addEventListener("pointerdown",he,{passive:!0}),window.addEventListener("pointermove",ve,{passive:!0});let Ke=!0,yt=()=>{Ke=!document.hidden};document.addEventListener("visibilitychange",yt);let Je=new ql,Z=Math.random()*1e3,ie=0,te=()=>{ie=requestAnimationFrame(te),!(f&&!Ke)&&(E.uTime.value=Z+Je.getElapsedTime()*g,k&&(k.uniforms.get("uTime").value=E.uTime.value),Y?(N?.update(),Y.render()):x.render(R,D))};return ie=requestAnimationFrame(te),()=>{cancelAnimationFrame(ie),j.disconnect(),window.removeEventListener("pointerdown",he),window.removeEventListener("pointermove",ve),document.removeEventListener("visibilitychange",yt),q.geometry.dispose(),L.dispose(),Y?.dispose(),x.dispose(),x.forceContextLoss(),x.domElement.parentElement===b&&b.removeChild(x.domElement)}},[r,f,n,_,u,l,h,c,m,o,a,e,p,d,M,v,g,S,t]),(0,YA.jsx)("div",{ref:T,className:`pixel-blast-container ${i}`,style:s,"aria-hidden":"true"})}var iv=Tr(lc());function SD(){let[t,e]=(0,Gh.useState)(document.documentElement.dataset.theme==="light");return(0,Gh.useEffect)(()=>{let n=new MutationObserver(()=>{e(document.documentElement.dataset.theme==="light")});return n.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme"]}),()=>n.disconnect()},[]),(0,iv.jsx)(nv,{variant:"square",pixelSize:4,color:t?"#000000":"#ffffff",patternScale:2,patternDensity:1.4,pixelSizeJitter:.5,enableRipples:!0,rippleSpeed:.4,rippleThickness:.12,rippleIntensityScale:1.5,liquid:!1,speed:.5,edgeFade:0,transparent:!0})}var qA=document.getElementById("pixel-blast-root");qA&&(0,QA.createRoot)(qA).render((0,iv.jsx)(SD,{}));})();
/*! Bundled license information:

react/cjs/react.production.js:
  (**
   * @license React
   * react.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

scheduler/cjs/scheduler.production.js:
  (**
   * @license React
   * scheduler.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react-dom/cjs/react-dom.production.js:
  (**
   * @license React
   * react-dom.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react-dom/cjs/react-dom-client.production.js:
  (**
   * @license React
   * react-dom-client.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

react/cjs/react-jsx-runtime.production.js:
  (**
   * @license React
   * react-jsx-runtime.production.js
   *
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   *
   * This source code is licensed under the MIT license found in the
   * LICENSE file in the root directory of this source tree.
   *)

three/build/three.core.js:
three/build/three.module.js:
  (**
   * @license
   * Copyright 2010-2026 Three.js Authors
   * SPDX-License-Identifier: MIT
   *)

postprocessing/build/index.js:
  (**
   * postprocessing v6.39.4 build Mon Jul 27 2026
   * https://github.com/pmndrs/postprocessing
   * Copyright 2015-2026 Raoul van Rüschen
   * @license Zlib
   *)
*/
