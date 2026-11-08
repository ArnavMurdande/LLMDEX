(()=>{var z1=Object.create;var h0=Object.defineProperty;var H1=Object.getOwnPropertyDescriptor;var G1=Object.getOwnPropertyNames;var V1=Object.getPrototypeOf,k1=Object.prototype.hasOwnProperty;var Zi=(t,e)=>()=>{try{return e||t((e={exports:{}}).exports,e),e.exports}catch(n){throw e=0,n}};var W1=(t,e,n,i)=>{if(e&&typeof e=="object"||typeof e=="function")for(let s of G1(e))!k1.call(t,s)&&s!==n&&h0(t,s,{get:()=>e[s],enumerable:!(i=H1(e,s))||i.enumerable});return t};var Wr=(t,e,n)=>(n=t!=null?z1(V1(t)):{},W1(e||!t||!t.__esModule?h0(n,"default",{value:t,enumerable:!0}):n,t));var T0=Zi(Pe=>{"use strict";var wd=Symbol.for("react.transitional.element"),X1=Symbol.for("react.portal"),Y1=Symbol.for("react.fragment"),q1=Symbol.for("react.strict_mode"),Q1=Symbol.for("react.profiler"),Z1=Symbol.for("react.consumer"),K1=Symbol.for("react.context"),J1=Symbol.for("react.forward_ref"),j1=Symbol.for("react.suspense"),$1=Symbol.for("react.memo"),v0=Symbol.for("react.lazy"),eE=Symbol.for("react.activity"),tE=Symbol.for("react.view_transition"),d0=Symbol.iterator;function nE(t){return t===null||typeof t!="object"?null:(t=d0&&t[d0]||t["@@iterator"],typeof t=="function"?t:null)}var x0={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},y0=Object.assign,_0={};function Ra(t,e,n){this.props=t,this.context=e,this.refs=_0,this.updater=n||x0}Ra.prototype.isReactComponent={};Ra.prototype.setState=function(t,e){if(typeof t!="object"&&typeof t!="function"&&t!=null)throw Error("takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,t,e,"setState")};Ra.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this,t,"forceUpdate")};function S0(){}S0.prototype=Ra.prototype;function Cd(t,e,n){this.props=t,this.context=e,this.refs=_0,this.updater=n||x0}var Rd=Cd.prototype=new S0;Rd.constructor=Cd;y0(Rd,Ra.prototype);Rd.isPureReactComponent=!0;var p0=Array.isArray;function bd(){}var _t={H:null,A:null,T:null,S:null},A0=Object.prototype.hasOwnProperty;function Dd(t,e,n){var i=n.ref;return{$$typeof:wd,type:t,key:e,ref:i!==void 0?i:null,props:n}}function iE(t,e){return Dd(t.type,e,t.props)}function Ud(t){return typeof t=="object"&&t!==null&&t.$$typeof===wd}function sE(t){var e={"=":"=0",":":"=2"};return"$"+t.replace(/[=:]/g,function(n){return e[n]})}var m0=/\/+/g;function Td(t,e){return typeof t=="object"&&t!==null&&t.key!=null?sE(""+t.key):e.toString(36)}function rE(t){switch(t.status){case"fulfilled":return t.value;case"rejected":throw t.reason;default:switch(typeof t.status=="string"?t.then(bd,bd):(t.status="pending",t.then(function(e){t.status==="pending"&&(t.status="fulfilled",t.value=e)},function(e){t.status==="pending"&&(t.status="rejected",t.reason=e)})),t.status){case"fulfilled":return t.value;case"rejected":throw t.reason}}throw t}function Ca(t,e,n,i,s){var r=typeof t;(r==="undefined"||r==="boolean")&&(t=null);var a=!1;if(t===null)a=!0;else switch(r){case"bigint":case"string":case"number":a=!0;break;case"object":switch(t.$$typeof){case wd:case X1:a=!0;break;case v0:return a=t._init,Ca(a(t._payload),e,n,i,s)}}if(a)return s=s(t),a=i===""?"."+Td(t,0):i,p0(s)?(n="",a!=null&&(n=a.replace(m0,"$&/")+"/"),Ca(s,e,n,"",function(c){return c})):s!=null&&(Ud(s)&&(s=iE(s,n+(s.key==null||t&&t.key===s.key?"":(""+s.key).replace(m0,"$&/")+"/")+a)),e.push(s)),1;a=0;var o=i===""?".":i+":";if(p0(t))for(var l=0;l<t.length;l++)i=t[l],r=o+Td(i,l),a+=Ca(i,e,n,r,s);else if(l=nE(t),typeof l=="function")for(t=l.call(t),l=0;!(i=t.next()).done;)i=i.value,r=o+Td(i,l++),a+=Ca(i,e,n,r,s);else if(r==="object"){if(typeof t.then=="function")return Ca(rE(t),e,n,i,s);throw e=String(t),Error("Objects are not valid as a React child (found: "+(e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e)+"). If you meant to render a collection of children, use an array instead.")}return a}function Gc(t,e,n){if(t==null)return t;var i=[],s=0;return Ca(t,i,"","",function(r){return e.call(n,r,s++)}),i}function aE(t){if(t._status===-1){var e=t._result,n=e();n.then(function(i){(t._status===0||t._status===-1)&&(t._status=1,t._result=i,n.status===void 0&&(n.status="fulfilled",n.value=i))},function(i){(t._status===0||t._status===-1)&&(t._status=2,t._result=i,n.status===void 0&&(n.status="rejected",n.reason=i))}),t._status===-1&&(t._status=0,t._result=n)}if(t._status===1)return t._result.default;throw t._result}var g0=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)};function M0(t){var e=_t.T,n={};n.types=e!==null?e.types:null,_t.T=n;try{var i=t(),s=_t.S;s!==null&&s(n,i),typeof i=="object"&&i!==null&&typeof i.then=="function"&&i.then(bd,g0)}catch(r){g0(r)}finally{e!==null&&n.types!==null&&(e.types=n.types),_t.T=e}}function E0(t){var e=_t.T;if(e!==null){var n=e.types;n===null?e.types=[t]:n.indexOf(t)===-1&&n.push(t)}else M0(E0.bind(null,t))}var oE={map:Gc,forEach:function(t,e,n){Gc(t,function(){e.apply(this,arguments)},n)},count:function(t){var e=0;return Gc(t,function(){e++}),e},toArray:function(t){return Gc(t,function(e){return e})||[]},only:function(t){if(!Ud(t))throw Error("React.Children.only expected to receive a single React element child.");return t}};Pe.Activity=eE;Pe.Children=oE;Pe.Component=Ra;Pe.Fragment=Y1;Pe.Profiler=Q1;Pe.PureComponent=Cd;Pe.StrictMode=q1;Pe.Suspense=j1;Pe.ViewTransition=tE;Pe.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=_t;Pe.__COMPILER_RUNTIME={__proto__:null,c:function(t){return _t.H.useMemoCache(t)}};Pe.addTransitionType=E0;Pe.cache=function(t){return function(){return t.apply(null,arguments)}};Pe.cacheSignal=function(){return null};Pe.cloneElement=function(t,e,n){if(t==null)throw Error("The argument must be a React element, but you passed "+t+".");var i=y0({},t.props),s=t.key;if(e!=null)for(r in e.key!==void 0&&(s=""+e.key),e)!A0.call(e,r)||r==="key"||r==="__self"||r==="__source"||r==="ref"&&e.ref===void 0||(i[r]=e[r]);var r=arguments.length-2;if(r===1)i.children=n;else if(1<r){for(var a=Array(r),o=0;o<r;o++)a[o]=arguments[o+2];i.children=a}return Dd(t.type,s,i)};Pe.createContext=function(t){return t={$$typeof:K1,_currentValue:t,_currentValue2:t,_threadCount:0,Provider:null,Consumer:null},t.Provider=t,t.Consumer={$$typeof:Z1,_context:t},t};Pe.createElement=function(t,e,n){var i,s={},r=null;if(e!=null)for(i in e.key!==void 0&&(r=""+e.key),e)A0.call(e,i)&&i!=="key"&&i!=="__self"&&i!=="__source"&&(s[i]=e[i]);var a=arguments.length-2;if(a===1)s.children=n;else if(1<a){for(var o=Array(a),l=0;l<a;l++)o[l]=arguments[l+2];s.children=o}if(t&&t.defaultProps)for(i in a=t.defaultProps,a)s[i]===void 0&&(s[i]=a[i]);return Dd(t,r,s)};Pe.createRef=function(){return{current:null}};Pe.forwardRef=function(t){return{$$typeof:J1,render:t}};Pe.isValidElement=Ud;Pe.lazy=function(t){return{$$typeof:v0,_payload:{_status:-1,_result:t},_init:aE}};Pe.memo=function(t,e){return{$$typeof:$1,type:t,compare:e===void 0?null:e}};Pe.startTransition=M0;Pe.unstable_useCacheRefresh=function(){return _t.H.useCacheRefresh()};Pe.use=function(t){return _t.H.use(t)};Pe.useActionState=function(t,e,n){return _t.H.useActionState(t,e,n)};Pe.useCallback=function(t,e){return _t.H.useCallback(t,e)};Pe.useContext=function(t){return _t.H.useContext(t)};Pe.useDebugValue=function(){};Pe.useDeferredValue=function(t,e){return _t.H.useDeferredValue(t,e)};Pe.useEffect=function(t,e){return _t.H.useEffect(t,e)};Pe.useEffectEvent=function(t){return _t.H.useEffectEvent(t)};Pe.useId=function(){return _t.H.useId()};Pe.useImperativeHandle=function(t,e,n){return _t.H.useImperativeHandle(t,e,n)};Pe.useInsertionEffect=function(t,e){return _t.H.useInsertionEffect(t,e)};Pe.useLayoutEffect=function(t,e){return _t.H.useLayoutEffect(t,e)};Pe.useMemo=function(t,e){return _t.H.useMemo(t,e)};Pe.useOptimistic=function(t,e){return _t.H.useOptimistic(t,e)};Pe.useReducer=function(t,e,n){return _t.H.useReducer(t,e,n)};Pe.useRef=function(t){return _t.H.useRef(t)};Pe.useState=function(t){return _t.H.useState(t)};Pe.useSyncExternalStore=function(t,e,n){return _t.H.useSyncExternalStore(t,e,n)};Pe.useTransition=function(){return _t.H.useTransition()};Pe.version="19.3.1"});var nl=Zi((kU,b0)=>{"use strict";b0.exports=T0()});var P0=Zi(Ct=>{"use strict";function Ld(t,e){var n=t.length;t.push(e);e:for(;0<n;){var i=n-1>>>1,s=t[i];if(0<Vc(s,e))t[i]=e,t[n]=s,n=i;else break e}}function Ki(t){return t.length===0?null:t[0]}function Wc(t){if(t.length===0)return null;var e=t[0],n=t.pop();if(n!==e){t[0]=n;e:for(var i=0,s=t.length,r=s>>>1;i<r;){var a=2*(i+1)-1,o=t[a],l=a+1,c=t[l];if(0>Vc(o,n))l<s&&0>Vc(c,o)?(t[i]=c,t[l]=n,i=l):(t[i]=o,t[a]=n,i=a);else if(l<s&&0>Vc(c,n))t[i]=c,t[l]=n,i=l;else break e}}return e}function Vc(t,e){var n=t.sortIndex-e.sortIndex;return n!==0?n:t.id-e.id}Ct.unstable_now=void 0;typeof performance=="object"&&typeof performance.now=="function"?(w0=performance,Ct.unstable_now=function(){return w0.now()}):(Id=Date,C0=Id.now(),Ct.unstable_now=function(){return Id.now()-C0});var w0,Id,C0,Ss=[],Zs=[],lE=1,vi=null,En=3,Pd=!1,il=!1,sl=!1,Od=!1,U0=typeof setTimeout=="function"?setTimeout:null,I0=typeof clearTimeout=="function"?clearTimeout:null,R0=typeof setImmediate<"u"?setImmediate:null;function kc(t){for(var e=Ki(Zs);e!==null;){if(e.callback===null)Wc(Zs);else if(e.startTime<=t)Wc(Zs),e.sortIndex=e.expirationTime,Ld(Ss,e);else break;e=Ki(Zs)}}function Fd(t){if(sl=!1,kc(t),!il)if(Ki(Ss)!==null)il=!0,Ua||(Ua=!0,Da());else{var e=Ki(Zs);e!==null&&zd(Fd,e.startTime-t)}}var Ua=!1,rl=-1,B0=5,N0=-1;function L0(){return Od?!0:!(Ct.unstable_now()-N0<B0)}function Bd(){if(Od=!1,Ua){var t=Ct.unstable_now();N0=t;var e=!0;try{e:{il=!1,sl&&(sl=!1,I0(rl),rl=-1),Pd=!0;var n=En;try{t:{for(kc(t),vi=Ki(Ss);vi!==null&&!(vi.expirationTime>t&&L0());){var i=vi.callback;if(typeof i=="function"){vi.callback=null,En=vi.priorityLevel;var s=i(vi.expirationTime<=t);if(t=Ct.unstable_now(),typeof s=="function"){vi.callback=s,kc(t),e=!0;break t}vi===Ki(Ss)&&Wc(Ss),kc(t)}else Wc(Ss);vi=Ki(Ss)}if(vi!==null)e=!0;else{var r=Ki(Zs);r!==null&&zd(Fd,r.startTime-t),e=!1}}break e}finally{vi=null,En=n,Pd=!1}e=void 0}}finally{e?Da():Ua=!1}}}var Da;typeof R0=="function"?Da=function(){R0(Bd)}:typeof MessageChannel<"u"?(Nd=new MessageChannel,D0=Nd.port2,Nd.port1.onmessage=Bd,Da=function(){D0.postMessage(null)}):Da=function(){U0(Bd,0)};var Nd,D0;function zd(t,e){rl=U0(function(){t(Ct.unstable_now())},e)}Ct.unstable_IdlePriority=5;Ct.unstable_ImmediatePriority=1;Ct.unstable_LowPriority=4;Ct.unstable_NormalPriority=3;Ct.unstable_Profiling=null;Ct.unstable_UserBlockingPriority=2;Ct.unstable_cancelCallback=function(t){t.callback=null};Ct.unstable_forceFrameRate=function(t){0>t||125<t?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):B0=0<t?Math.floor(1e3/t):5};Ct.unstable_getCurrentPriorityLevel=function(){return En};Ct.unstable_next=function(t){switch(En){case 1:case 2:case 3:var e=3;break;default:e=En}var n=En;En=e;try{return t()}finally{En=n}};Ct.unstable_requestPaint=function(){Od=!0};Ct.unstable_runWithPriority=function(t,e){switch(t){case 1:case 2:case 3:case 4:case 5:break;default:t=3}var n=En;En=t;try{return e()}finally{En=n}};Ct.unstable_scheduleCallback=function(t,e,n){var i=Ct.unstable_now();switch(typeof n=="object"&&n!==null?(n=n.delay,n=typeof n=="number"&&0<n?i+n:i):n=i,t){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=n+s,t={id:lE++,callback:e,priorityLevel:t,startTime:n,expirationTime:s,sortIndex:-1},n>i?(t.sortIndex=n,Ld(Zs,t),Ki(Ss)===null&&t===Ki(Zs)&&(sl?(I0(rl),rl=-1):sl=!0,zd(Fd,n-i))):(t.sortIndex=s,Ld(Ss,t),il||Pd||(il=!0,Ua||(Ua=!0,Da()))),t};Ct.unstable_shouldYield=L0;Ct.unstable_wrapCallback=function(t){var e=En;return function(){var n=En;En=e;try{return t.apply(this,arguments)}finally{En=n}}}});var F0=Zi((XU,O0)=>{"use strict";O0.exports=P0()});var G0=Zi(Tn=>{"use strict";var cE=nl();function H0(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function Ks(){}var In={d:{f:Ks,r:function(){throw Error(H0(522))},D:Ks,C:Ks,L:Ks,m:Ks,X:Ks,S:Ks,M:Ks},p:0,findDOMNode:null},uE=Symbol.for("react.portal"),fE=Symbol.for("react.recoverable"),z0=Symbol.for("react.optimistic_key");function hE(t,e,n){var i=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:uE,key:i==null?null:i===z0?z0:""+i,children:t,containerInfo:e,implementation:n}}var al=cE.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function Xc(t,e){if(t==="font")return"";if(typeof e=="string")return e==="use-credentials"?e:""}Tn.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=In;Tn.browser=function(t){return{$$typeof:fE,_reason:t}};Tn.createPortal=function(t,e){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)throw Error(H0(299));return hE(t,e,null,n)};Tn.flushSync=function(t){var e=al.T,n=In.p;try{if(al.T=null,In.p=2,t)return t()}finally{al.T=e,In.p=n,In.d.f()}};Tn.preconnect=function(t,e){typeof t=="string"&&(e?(e=e.crossOrigin,e=typeof e=="string"?e==="use-credentials"?e:"":void 0):e=null,In.d.C(t,e))};Tn.prefetchDNS=function(t){typeof t=="string"&&In.d.D(t)};Tn.preinit=function(t,e){if(typeof t=="string"&&e&&typeof e.as=="string"){var n=e.as,i=Xc(n,e.crossOrigin),s=typeof e.integrity=="string"?e.integrity:void 0,r=typeof e.fetchPriority=="string"?e.fetchPriority:void 0;n==="style"?In.d.S(t,typeof e.precedence=="string"?e.precedence:void 0,{crossOrigin:i,integrity:s,fetchPriority:r}):n==="script"&&In.d.X(t,{crossOrigin:i,integrity:s,fetchPriority:r,nonce:typeof e.nonce=="string"?e.nonce:void 0})}};Tn.preinitModule=function(t,e){if(typeof t=="string")if(typeof e=="object"&&e!==null){if(e.as==null||e.as==="script"){var n=Xc(e.as,e.crossOrigin);In.d.M(t,{crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0})}}else e==null&&In.d.M(t)};Tn.preload=function(t,e){if(typeof t=="string"&&typeof e=="object"&&e!==null&&typeof e.as=="string"){var n=e.as,i=Xc(n,e.crossOrigin);In.d.L(t,n,{crossOrigin:i,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,type:typeof e.type=="string"?e.type:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0,referrerPolicy:typeof e.referrerPolicy=="string"?e.referrerPolicy:void 0,imageSrcSet:typeof e.imageSrcSet=="string"?e.imageSrcSet:void 0,imageSizes:typeof e.imageSizes=="string"?e.imageSizes:void 0,media:typeof e.media=="string"?e.media:void 0})}};Tn.preloadModule=function(t,e){if(typeof t=="string")if(e){var n=Xc(e.as,e.crossOrigin);In.d.m(t,{as:typeof e.as=="string"&&e.as!=="script"?e.as:void 0,crossOrigin:n,integrity:typeof e.integrity=="string"?e.integrity:void 0,nonce:typeof e.nonce=="string"?e.nonce:void 0,fetchPriority:typeof e.fetchPriority=="string"?e.fetchPriority:void 0})}else In.d.m(t)};Tn.requestFormReset=function(t){In.d.r(t)};Tn.unstable_batchedUpdates=function(t,e){return t(e)};Tn.useFormState=function(t,e,n){return al.H.useFormState(t,e,n)};Tn.useFormStatus=function(){return al.H.useHostTransitionStatus()};Tn.version="19.3.1"});var W0=Zi((qU,k0)=>{"use strict";function V0(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(V0)}catch(t){console.error(t)}}V0(),k0.exports=G0()});var DA=Zi(wf=>{"use strict";var $t=F0(),Ry=nl(),dE=W0();function J(t){var e="https://react.dev/errors/"+t;if(1<arguments.length){e+="?args[]="+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)e+="&args[]="+encodeURIComponent(arguments[n])}return"Minified React error #"+t+"; visit "+e+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}function Dy(t){return!(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)}function ql(t){for(var e=t,n=e;n&&!n.alternate;)e=n,(e.flags&4098)!==0&&(t=e.return),n=e.return;for(;e.return;)e=e.return;return e.tag===3?t:null}function Uy(t){if(t.tag===13){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function Iy(t){if(t.tag===31){var e=t.memoizedState;if(e===null&&(t=t.alternate,t!==null&&(e=t.memoizedState)),e!==null)return e.dehydrated}return null}function X0(t){if(ql(t)!==t)throw Error(J(188))}function pE(t){var e=t.alternate;if(!e){if(e=ql(t),e===null)throw Error(J(188));return e!==t?null:t}for(var n=t,i=e;;){var s=n.return;if(s===null)break;var r=s.alternate;if(r===null){if(i=s.return,i!==null){n=i;continue}break}if(s.child===r.child){for(r=s.child;r;){if(r===n)return X0(s),t;if(r===i)return X0(s),e;r=r.sibling}throw Error(J(188))}if(n.return!==i.return)n=s,i=r;else{for(var a=!1,o=s.child;o;){if(o===n){a=!0,n=s,i=r;break}if(o===i){a=!0,i=s,n=r;break}o=o.sibling}if(!a){for(o=r.child;o;){if(o===n){a=!0,n=r,i=s;break}if(o===i){a=!0,i=r,n=s;break}o=o.sibling}if(!a)throw Error(J(189))}}if(n.alternate!==i)throw Error(J(190))}if(n.tag!==3)throw Error(J(188));return n.stateNode.current===n?t:e}function By(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t;for(t=t.child;t!==null;){if(e=By(t),e!==null)return e;t=t.sibling}return null}function Xn(t,e,n,i,s,r){for(;t!==null;){if((t.tag===5||t.tag===27||t.tag===6)&&n(t,i,s,r)||(t.tag!==22||t.memoizedState===null)&&(e||t.tag!==5&&t.tag!==27)&&Xn(t.child,e,n,i,s,r))return!0;t=t.sibling}return!1}function fa(t){for(t=t.return;t!==null;){if(t.tag===3||t.tag===5||t.tag===27)return t;t=t.return}return null}function Y0(t){var e=!1;for(t=t.return;t!==null&&(t.tag===4&&(e=!0),!(t.tag===3||t.tag===5||t.tag===27));)t=t.return;return e}function Ny(t){var e=[null,null],n=fa(t);return n===null||Ly(e,t,n.child,{foundSelf:!1}),e}function Ly(t,e,n,i){for(;n!==null;){if(n===e)i.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(i.foundSelf)return t[1]=n,!0;t[0]=n}else if((n.tag!==22||n.memoizedState===null)&&Ly(t,e,n.child,i))return!0;n=n.sibling}return!1}function jt(t){switch(t.tag){case 5:case 27:case 6:return t.stateNode;case 3:return t.stateNode.containerInfo;default:throw Error(J(559))}}var Fa=null,vp=null;function mE(t,e,n){return t===n?!0:t===e?(Fa=t,!0):!1}function gE(t,e,n){return t===n?(vp=t,!1):t===e?(vp!==null&&(Fa=t),!0):!1}function q0(t){if(t===null)return null;do t=t===null?null:t.return;while(t&&t.tag!==5&&t.tag!==27&&t.tag!==3);return t||null}function xp(t,e,n){for(var i=0,s=t;s;s=n(s))i++;s=0;for(var r=e;r;r=n(r))s++;for(;0<i-s;)t=n(t),i--;for(;0<s-i;)e=n(e),s--;for(;i--;){if(t===e||e!==null&&t===e.alternate)return t;t=n(t),e=n(e)}return null}var xt=Object.assign,vE=Symbol.for("react.element"),Yc=Symbol.for("react.transitional.element"),dl=Symbol.for("react.portal"),za=Symbol.for("react.fragment"),Py=Symbol.for("react.strict_mode"),yp=Symbol.for("react.profiler"),Oy=Symbol.for("react.consumer"),ns=Symbol.for("react.context"),Cm=Symbol.for("react.forward_ref"),_p=Symbol.for("react.suspense"),Sp=Symbol.for("react.suspense_list"),Rm=Symbol.for("react.memo"),er=Symbol.for("react.lazy"),Ap=Symbol.for("react.activity"),xE=Symbol.for("react.legacy_hidden"),yE=Symbol.for("react.memo_cache_sentinel"),Mp=Symbol.for("react.view_transition"),_E=Symbol.for("react.recoverable"),Q0=Symbol.iterator;function ol(t){return t===null||typeof t!="object"?null:(t=Q0&&t[Q0]||t["@@iterator"],typeof t=="function"?t:null)}var SE=Symbol.for("react.client.reference");function Ep(t){if(t==null)return null;if(typeof t=="function")return t.$$typeof===SE?null:t.displayName||t.name||null;if(typeof t=="string")return t;switch(t){case za:return"Fragment";case yp:return"Profiler";case Py:return"StrictMode";case _p:return"Suspense";case Sp:return"SuspenseList";case Ap:return"Activity";case Mp:return"ViewTransition"}if(typeof t=="object")switch(t.$$typeof){case dl:return"Portal";case ns:return t.displayName||"Context";case Oy:return(t._context.displayName||"Context")+".Consumer";case Cm:var e=t.render;return t=t.displayName,t||(t=e.displayName||e.name||"",t=t!==""?"ForwardRef("+t+")":"ForwardRef"),t;case Rm:return e=t.displayName||null,e!==null?e:Ep(t.type)||"Memo";case er:e=t._payload,t=t._init;try{return Ep(t(e))}catch{}}return null}var pl=Array.isArray,Ue=Ry.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,ot=dE.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,jr={pending:!1,data:null,method:null,action:null},Tp=[],Ha=-1;function cs(t){return{current:t}}function xn(t){0>Ha||(t.current=Tp[Ha],Tp[Ha]=null,Ha--)}function Mt(t,e){Ha++,Tp[Ha]=t.current,t.current=e}var as=cs(null),Ul=cs(null),cr=cs(null),Iu=cs(null);function Bu(t,e){switch(Mt(cr,e),Mt(Ul,t),Mt(as,null),e.nodeType){case 9:case 11:t=(t=e.documentElement)&&(t=t.namespaceURI)?ly(t):0;break;default:if(t=e.tagName,e=e.namespaceURI)e=ly(e),t=rA(e,t);else switch(t){case"svg":t=1;break;case"math":t=2;break;default:t=0}}xn(as),Mt(as,t)}function ao(){xn(as),xn(Ul),xn(cr)}function bp(t){var e=t.memoizedState;e!==null&&(vo._currentValue=e.memoizedState,Mt(Iu,t)),e=as.current;var n=rA(e,t.type);e!==n&&(Mt(Ul,t),Mt(as,n))}function Nu(t){Ul.current===t&&(xn(as),xn(Ul)),Iu.current===t&&(xn(Iu),vo._currentValue=jr)}var Hd,Z0;function js(t){if(Hd===void 0)try{throw Error()}catch(n){var e=n.stack.trim().match(/\n( *(at )?)/);Hd=e&&e[1]||"",Z0=-1<n.stack.indexOf(`
    at`)?" (<anonymous>)":-1<n.stack.indexOf("@")?"@unknown:0:0":""}return`
`+Hd+t+Z0}var Gd=!1;function Vd(t,e){if(!t||Gd)return"";Gd=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var i={DetermineComponentFrameRoot:function(){try{if(e){var p=function(){throw Error()};if(Object.defineProperty(p.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(p,[])}catch(x){var u=x}Reflect.construct(t,[],p)}else{try{p.call()}catch(x){u=x}p=!1;try{var d=Object.getOwnPropertyDescriptor(t.prototype,"props");Object.defineProperty(t.prototype,"props",{configurable:!0,set:function(){throw Error()}}),p=!0,new t}finally{p&&(d!==void 0?Object.defineProperty(t.prototype,"props",d):delete t.prototype.props)}}}else{try{throw Error()}catch(x){u=x}(p=t())&&typeof p.catch=="function"&&p.catch(function(){})}}catch(x){if(x&&u&&typeof x.stack=="string")return[x.stack,u.stack]}return[null,null]}};i.DetermineComponentFrameRoot.displayName="DetermineComponentFrameRoot";var s=Object.getOwnPropertyDescriptor(i.DetermineComponentFrameRoot,"name");s&&s.configurable&&Object.defineProperty(i.DetermineComponentFrameRoot,"name",{value:"DetermineComponentFrameRoot"});var r=i.DetermineComponentFrameRoot(),a=r[0],o=r[1];if(a&&o){var l=a.split(`
`),c=o.split(`
`);for(s=i=0;i<l.length&&!l[i].includes("DetermineComponentFrameRoot");)i++;for(;s<c.length&&!c[s].includes("DetermineComponentFrameRoot");)s++;if(i===l.length||s===c.length)for(i=l.length-1,s=c.length-1;1<=i&&0<=s&&l[i]!==c[s];)s--;for(;1<=i&&0<=s;i--,s--)if(l[i]!==c[s]){if(i!==1||s!==1)do if(i--,s--,0>s||l[i]!==c[s]){var h=`
`+l[i].replace(" at new "," at ");return t.displayName&&h.includes("<anonymous>")&&(h=h.replace("<anonymous>",t.displayName)),h}while(1<=i&&0<=s);break}}}finally{Gd=!1,Error.prepareStackTrace=n}return(n=t?t.displayName||t.name:"")?js(n):""}function AE(t,e){switch(t.tag){case 26:case 27:case 5:return js(t.type);case 16:return js("Lazy");case 13:return t.child!==e&&e!==null?js("Suspense Fallback"):js("Suspense");case 19:return js("SuspenseList");case 0:case 15:return Vd(t.type,!1);case 11:return Vd(t.type.render,!1);case 1:return Vd(t.type,!0);case 31:return js("Activity");case 30:return js("ViewTransition");default:return""}}function K0(t){try{var e="",n=null;do e+=AE(t,n),n=t,t=t.return;while(t);return e}catch(i){return`
Error generating stack: `+i.message+`
`+i.stack}}var wp=Object.prototype.hasOwnProperty,Dm=$t.unstable_scheduleCallback,kd=$t.unstable_cancelCallback,ME=$t.unstable_shouldYield,EE=$t.unstable_requestPaint,ei=$t.unstable_now,TE=$t.unstable_getCurrentPriorityLevel,Fy=$t.unstable_ImmediatePriority,zy=$t.unstable_UserBlockingPriority,Lu=$t.unstable_NormalPriority,bE=$t.unstable_LowPriority,Hy=$t.unstable_IdlePriority,wE=$t.log,CE=$t.unstable_setDisableYieldValue,Ql=null,ti=null;function ir(t){if(typeof wE=="function"&&CE(t),ti&&typeof ti.setStrictMode=="function")try{ti.setStrictMode(Ql,t)}catch{}}var ni=Math.clz32?Math.clz32:UE,RE=Math.log,DE=Math.LN2;function UE(t){return t>>>=0,t===0?32:31-(RE(t)/DE|0)|0}var qc=256,Qc=262144,Zc=4194304;function qr(t){var e=t&42;if(e!==0)return e;switch(t&-t){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return t&-t;case 262144:case 524288:case 1048576:case 2097152:return t&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return t&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return t}}function lf(t,e,n){var i=t.pendingLanes;if(i===0)return 0;var s=0,r=t.suspendedLanes,a=t.pingedLanes;t=t.warmLanes;var o=i&134217727;return o!==0?(i=o&~r,i!==0?s=qr(i):(a&=o,a!==0?s=qr(a):n||(n=o&~t,n!==0&&(s=qr(n))))):(o=i&~r,o!==0?s=qr(o):a!==0?s=qr(a):n||(n=i&~t,n!==0&&(s=qr(n)))),s===0?0:e!==0&&e!==s&&(e&r)===0&&(r=s&-s,n=e&-e,r>=n||r===32&&(n&4194048)!==0)?e:s}function Zl(t,e){return(t.pendingLanes&~(t.suspendedLanes&~t.pingedLanes)&e)===0}function Gy(t,e){(e&8)!==0&&(e|=e&32);var n=t.entangledLanes;if(n!==0)for(t=t.entanglements,n&=e;0<n;){var i=31-ni(n),s=1<<i;e|=t[i],n&=~s}return e}function IE(t,e){switch(t){case 1:case 2:case 4:case 8:case 64:return e+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Vy(){var t=Zc;return Zc<<=1,(Zc&62914560)===0&&(Zc=4194304),t}function Wd(t){for(var e=[],n=0;31>n;n++)e.push(t);return e}function Kl(t,e){t.pendingLanes|=e,e!==268435456&&(t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0)}function BE(t,e,n,i,s,r){var a=t.pendingLanes;t.pendingLanes=n,t.suspendedLanes=0,t.pingedLanes=0,t.warmLanes=0,t.expiredLanes&=n,t.entangledLanes&=n,t.errorRecoveryDisabledLanes&=n,t.shellSuspendCounter=0;var o=t.entanglements,l=t.expirationTimes,c=t.hiddenUpdates;for(n=a&~n;0<n;){var h=31-ni(n),p=1<<h;o[h]=0,l[h]=-1;var u=c[h];if(u!==null)for(c[h]=null,h=0;h<u.length;h++){var d=u[h];d!==null&&(d.lane&=-536870913)}n&=~p}i!==0&&ky(t,i,0),r!==0&&s===0&&t.tag!==0&&(t.suspendedLanes|=r&~(a&~e))}function ky(t,e,n){t.pendingLanes|=e,t.suspendedLanes&=~e;var i=31-ni(e);t.entangledLanes|=e,t.entanglements[i]=t.entanglements[i]|1073741824|n&261930}function Wy(t,e){var n=t.entangledLanes|=e;for(t=t.entanglements;n;){var i=31-ni(n),s=1<<i;s&e|t[i]&e&&(t[i]|=e),n&=~s}}function Xy(t,e){var n=e&-e;return n=(n&42)!==0?1:Um(n),(n&(t.suspendedLanes|e))!==0?0:n}function Um(t){switch(t){case 2:t=1;break;case 8:t=4;break;case 32:t=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:t=128;break;case 268435456:t=134217728;break;default:t=0}return t}function Im(t){return t&=-t,2<t?8<t?(t&134217727)!==0?32:268435456:8:2}function Yy(){var t=ot.p;return t!==0?t:(t=window.event,t===void 0?32:wA(t.type))}function J0(t,e){var n=ot.p;try{return ot.p=t,e()}finally{ot.p=n}}var Ns=Math.random().toString(36).slice(2),gn="__reactFiber$"+Ns,Yn="__reactProps$"+Ns,_o="__reactContainer$"+Ns,j0="__reactEvents$"+Ns,NE="__reactListeners$"+Ns,LE="__reactHandles$"+Ns,$0="__reactResources$"+Ns,Jl="__reactMarker$"+Ns,Pu="__reactLoad$"+Ns;function cf(t){delete t[gn],delete t[Yn],delete t[NE],delete t[LE]}function Kr(t){var e;if(e=t[gn])return e;for(var n=t.parentNode;n;){if(e=n[_o]||n[gn]){if(n=e.alternate,e.child!==null||n!==null&&n.child!==null)for(t=gy(t);t!==null;){if(n=t[gn])return n;t=gy(t)}return e}t=n,n=t.parentNode}return null}function So(t){if(t=t[gn]||t[_o]){var e=t.tag;if(e===5||e===6||e===13||e===31||e===26||e===27||e===3)return t}return null}function ml(t){var e=t.tag;if(e===5||e===26||e===27||e===6)return t.stateNode;throw Error(J(33))}function Ka(t){var e=t[$0];return e||(e=t[$0]={hoistableStyles:new Map,hoistableScripts:new Map}),e}function ln(t){t[Jl]=!0}function qy(t){t[Pu]=void 0}var Qy=new Set,Zy={};function ha(t,e){oo(t,e),oo(t+"Capture",e)}function oo(t,e){for(Zy[t]=e,t=0;t<e.length;t++)Qy.add(e[t])}var PE=RegExp("^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$"),ex={},tx={};function OE(t){return wp.call(tx,t)?!0:wp.call(ex,t)?!1:PE.test(t)?tx[t]=!0:(ex[t]=!0,!1)}var nt=!1;function nx(){var t=nt;return nt=!1,t}function du(t,e,n){if(OE(e))if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":t.removeAttribute(e);return;case"boolean":var i=e.toLowerCase().slice(0,5);if(i!=="data-"&&i!=="aria-"){t.removeAttribute(e);return}}t.setAttribute(e,n)}}function Kc(t,e,n){if(n===null)t.removeAttribute(e);else{switch(typeof n){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(e);return}t.setAttribute(e,n)}}function As(t,e,n,i){if(i===null)t.removeAttribute(n);else{switch(typeof i){case"undefined":case"function":case"symbol":case"boolean":t.removeAttribute(n);return}t.setAttributeNS(e,n,i)}}function Kn(t){switch(typeof t){case"bigint":case"boolean":case"number":case"string":case"undefined":return t;case"object":return t;default:return""}}function Ky(t){var e=t.type;return(t=t.nodeName)&&t.toLowerCase()==="input"&&(e==="checkbox"||e==="radio")}function FE(t,e,n){var i=Object.getOwnPropertyDescriptor(t.constructor.prototype,e);if(!t.hasOwnProperty(e)&&typeof i<"u"&&typeof i.get=="function"&&typeof i.set=="function"){var s=i.get,r=i.set;return Object.defineProperty(t,e,{configurable:!0,get:function(){return s.call(this)},set:function(a){n=""+a,r.call(this,a)}}),Object.defineProperty(t,e,{enumerable:i.enumerable}),{getValue:function(){return n},setValue:function(a){n=""+a},stopTracking:function(){t._valueTracker=null,delete t[e]}}}}function Cp(t){if(!t._valueTracker){var e=Ky(t)?"checked":"value";t._valueTracker=FE(t,e,""+t[e])}}function Jy(t){if(!t)return!1;var e=t._valueTracker;if(!e)return!0;var n=e.getValue(),i="";return t&&(i=Ky(t)?t.checked?"true":"false":t.value),t=i,t!==n?(e.setValue(t),!0):!1}var zE=/[\n"\\]/g;function Ai(t){return t.replace(zE,function(e){return"\\"+e.charCodeAt(0).toString(16)+" "})}function Rp(t,e,n,i,s,r,a,o){t.name="",a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"?t.type=a:t.removeAttribute("type"),e!=null?a==="number"?(e===0&&t.value===""||t.value!=e)&&(t.value=""+Kn(e)):t.value!==""+Kn(e)&&(t.value=""+Kn(e)):a!=="submit"&&a!=="reset"||t.removeAttribute("value"),e!=null?a==="number"&&t.value==e?Xd(t,Kn(t.value)):Xd(t,Kn(e)):n!=null?Xd(t,Kn(n)):i!=null&&t.removeAttribute("value"),s==null&&r!=null&&(t.defaultChecked=!!r),s!=null&&(t.checked=s&&typeof s!="function"&&typeof s!="symbol"),o!=null&&typeof o!="function"&&typeof o!="symbol"&&typeof o!="boolean"?t.name=""+Kn(o):t.removeAttribute("name")}function jy(t,e,n,i,s,r,a,o){if(r!=null&&typeof r!="function"&&typeof r!="symbol"&&typeof r!="boolean"&&(t.type=r),e!=null||n!=null){if(!(r!=="submit"&&r!=="reset"||e!=null)){Cp(t);return}n=n!=null?""+Kn(n):"",e=e!=null?""+Kn(e):n,o||e===t.value||(t.value=e),t.defaultValue=e}i=i??s,i=typeof i!="function"&&typeof i!="symbol"&&!!i,t.checked=o?t.checked:!!i,t.defaultChecked=!!i,a!=null&&typeof a!="function"&&typeof a!="symbol"&&typeof a!="boolean"&&(t.name=a),Cp(t)}function Xd(t,e){t.defaultValue!==""+e&&(t.defaultValue=""+e)}function Ja(t,e,n,i){if(t=t.options,e){e={};for(var s=0;s<n.length;s++)e["$"+n[s]]=!0;for(n=0;n<t.length;n++)s=e.hasOwnProperty("$"+t[n].value),t[n].selected!==s&&(t[n].selected=s),s&&i&&(t[n].defaultSelected=!0)}else{for(n=""+Kn(n),e=null,s=0;s<t.length;s++){if(t[s].value===n){t[s].selected=!0,i&&(t[s].defaultSelected=!0);return}e!==null||t[s].disabled||(e=t[s])}e!==null&&(e.selected=!0)}}function $y(t,e,n){if(e!=null&&(e=""+Kn(e),e!==t.value&&(t.value=e),n==null)){t.defaultValue!==e&&(t.defaultValue=e);return}t.defaultValue=n!=null?""+Kn(n):""}function e_(t,e,n,i){if(e==null){if(i!=null){if(n!=null)throw Error(J(92));if(pl(i)){if(1<i.length)throw Error(J(93));i=i[0]}n=i}n==null&&(n=""),e=n}n=Kn(e),t.defaultValue=n,i=t.textContent,i===n&&i!==""&&i!==null&&(t.value=i),Cp(t)}function lo(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&n.nodeType===3){n.nodeValue=e;return}}t.textContent=e}var HE=new Set("animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp".split(" "));function ix(t,e,n){var i=e.indexOf("--")===0;n==null||typeof n=="boolean"||n===""?i?t.setProperty(e,""):e==="float"?t.cssFloat="":t[e]="":i?t.setProperty(e,n):typeof n!="number"||n===0||HE.has(e)?e==="float"?t.cssFloat=n:t[e]=(""+n).trim():t[e]=n+"px"}function t_(t,e,n){if(e!=null&&typeof e!="object")throw Error(J(62));if(t=t.style,n!=null){for(var i in n)!n.hasOwnProperty(i)||e!=null&&e.hasOwnProperty(i)||(i.indexOf("--")===0?t.setProperty(i,""):i==="float"?t.cssFloat="":t[i]="",nt=!0);for(var s in e)i=e[s],e.hasOwnProperty(s)&&n[s]!==i&&(ix(t,s,i),nt=!0)}else for(var r in e)e.hasOwnProperty(r)&&ix(t,r,e[r])}function Bm(t){if(t.indexOf("-")===-1)return!1;switch(t){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var GE=new Map([["acceptCharset","accept-charset"],["htmlFor","for"],["httpEquiv","http-equiv"],["crossOrigin","crossorigin"],["accentHeight","accent-height"],["alignmentBaseline","alignment-baseline"],["arabicForm","arabic-form"],["baselineShift","baseline-shift"],["capHeight","cap-height"],["clipPath","clip-path"],["clipRule","clip-rule"],["colorInterpolation","color-interpolation"],["colorInterpolationFilters","color-interpolation-filters"],["colorProfile","color-profile"],["colorRendering","color-rendering"],["dominantBaseline","dominant-baseline"],["enableBackground","enable-background"],["fillOpacity","fill-opacity"],["fillRule","fill-rule"],["floodColor","flood-color"],["floodOpacity","flood-opacity"],["fontFamily","font-family"],["fontSize","font-size"],["fontSizeAdjust","font-size-adjust"],["fontStretch","font-stretch"],["fontStyle","font-style"],["fontVariant","font-variant"],["fontWeight","font-weight"],["glyphName","glyph-name"],["glyphOrientationHorizontal","glyph-orientation-horizontal"],["glyphOrientationVertical","glyph-orientation-vertical"],["horizAdvX","horiz-adv-x"],["horizOriginX","horiz-origin-x"],["imageRendering","image-rendering"],["letterSpacing","letter-spacing"],["lightingColor","lighting-color"],["markerEnd","marker-end"],["markerMid","marker-mid"],["markerStart","marker-start"],["maskType","mask-type"],["overlinePosition","overline-position"],["overlineThickness","overline-thickness"],["paintOrder","paint-order"],["panose-1","panose-1"],["pointerEvents","pointer-events"],["renderingIntent","rendering-intent"],["shapeRendering","shape-rendering"],["stopColor","stop-color"],["stopOpacity","stop-opacity"],["strikethroughPosition","strikethrough-position"],["strikethroughThickness","strikethrough-thickness"],["strokeDasharray","stroke-dasharray"],["strokeDashoffset","stroke-dashoffset"],["strokeLinecap","stroke-linecap"],["strokeLinejoin","stroke-linejoin"],["strokeMiterlimit","stroke-miterlimit"],["strokeOpacity","stroke-opacity"],["strokeWidth","stroke-width"],["textAnchor","text-anchor"],["textDecoration","text-decoration"],["textRendering","text-rendering"],["transformOrigin","transform-origin"],["underlinePosition","underline-position"],["underlineThickness","underline-thickness"],["unicodeBidi","unicode-bidi"],["unicodeRange","unicode-range"],["unitsPerEm","units-per-em"],["vAlphabetic","v-alphabetic"],["vHanging","v-hanging"],["vIdeographic","v-ideographic"],["vMathematical","v-mathematical"],["vectorEffect","vector-effect"],["vertAdvY","vert-adv-y"],["vertOriginX","vert-origin-x"],["vertOriginY","vert-origin-y"],["wordSpacing","word-spacing"],["writingMode","writing-mode"],["xmlnsXlink","xmlns:xlink"],["xHeight","x-height"]]),VE=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function pu(t){return VE.test(""+t)?"javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')":t}function is(){}var Dp=null;function Nm(t){return t=t.target||t.srcElement||window,t.correspondingUseElement&&(t=t.correspondingUseElement),t.nodeType===3?t.parentNode:t}var Ga=null,ja=null;function sx(t){var e=So(t);if(e&&(t=e.stateNode)){var n=t[Yn]||null;e:switch(t=e.stateNode,e.type){case"input":if(Rp(t,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),e=n.name,n.type==="radio"&&e!=null){for(n=t;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll('input[name="'+Ai(""+e)+'"][type="radio"]'),e=0;e<n.length;e++){var i=n[e];if(i!==t&&i.form===t.form){var s=i[Yn]||null;if(!s)throw Error(J(90));Rp(i,s.value,s.defaultValue,s.defaultValue,s.checked,s.defaultChecked,s.type,s.name)}}for(e=0;e<n.length;e++)i=n[e],i.form===t.form&&Jy(i)}break e;case"textarea":$y(t,n.value,n.defaultValue);break e;case"select":e=n.value,e!=null&&Ja(t,!!n.multiple,e,!1)}}}var Yd=!1;function n_(t,e,n){if(Yd)return t(e,n);Yd=!0;try{var i=t(e);return i}finally{if(Yd=!1,(Ga!==null||ja!==null)&&(Mf(),Ga&&(e=Ga,t=ja,ja=Ga=null,sx(e),t)))for(e=0;e<t.length;e++)sx(t[e])}}function Il(t,e){var n=t.stateNode;if(n===null)return null;var i=n[Yn]||null;if(i===null)return null;n=i[e];e:switch(e){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(i=!i.disabled)||(t=t.type,i=!(t==="button"||t==="input"||t==="select"||t==="textarea")),t=!i;break e;default:t=!1}if(t)return null;if(n&&typeof n!="function")throw Error(J(231,e,typeof n));return n}var Cs=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Up=!1;if(Cs)try{Ia={},Object.defineProperty(Ia,"passive",{get:function(){Up=!0}}),window.addEventListener("test",Ia,Ia),window.removeEventListener("test",Ia,Ia)}catch{Up=!1}var Ia,sr=null,Lm=null,mu=null;function i_(){if(mu)return mu;var t,e=Lm,n=e.length,i,s="value"in sr?sr.value:sr.textContent,r=s.length;for(t=0;t<n&&e[t]===s[t];t++);var a=n-t;for(i=1;i<=a&&e[n-i]===s[r-i];i++);return mu=s.slice(t,1<i?1-i:void 0)}function gu(t){var e=t.keyCode;return"charCode"in t?(t=t.charCode,t===0&&e===13&&(t=13)):t=e,t===10&&(t=13),32<=t||t===13?t:0}function Jc(){return!0}function rx(){return!1}function Pn(t){function e(n,i,s,r,a){this._reactName=n,this._targetInst=s,this.type=i,this.nativeEvent=r,this.target=a,this.currentTarget=null;for(var o in t)t.hasOwnProperty(o)&&(n=t[o],this[o]=n?n(r):r[o]);return this.isDefaultPrevented=(r.defaultPrevented!=null?r.defaultPrevented:r.returnValue===!1)?Jc:rx,this.isPropagationStopped=rx,this}return xt(e.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Jc)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Jc)},persist:function(){},isPersistent:Jc}),e}var Er={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},uf=Pn(Er),jl=xt({},Er,{view:0,detail:0}),kE=Pn(jl),qd,Qd,ll,ff=xt({},jl,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Pm,button:0,buttons:0,relatedTarget:function(t){return t.relatedTarget===void 0?t.fromElement===t.srcElement?t.toElement:t.fromElement:t.relatedTarget},movementX:function(t){return"movementX"in t?t.movementX:(t!==ll&&(ll&&t.type==="mousemove"?(qd=t.screenX-ll.screenX,Qd=t.screenY-ll.screenY):Qd=qd=0,ll=t),qd)},movementY:function(t){return"movementY"in t?t.movementY:Qd}}),ax=Pn(ff),WE=xt({},ff,{dataTransfer:0}),XE=Pn(WE),YE=xt({},jl,{relatedTarget:0}),Zd=Pn(YE),qE=xt({},Er,{animationName:0,elapsedTime:0,pseudoElement:0}),QE=Pn(qE),ZE=xt({},Er,{clipboardData:function(t){return"clipboardData"in t?t.clipboardData:window.clipboardData}}),KE=Pn(ZE),JE=xt({},Er,{data:0}),ox=Pn(JE),jE={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},$E={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},eT={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function tT(t){var e=this.nativeEvent;return e.getModifierState?e.getModifierState(t):(t=eT[t])?!!e[t]:!1}function Pm(){return tT}var nT=xt({},jl,{key:function(t){if(t.key){var e=jE[t.key]||t.key;if(e!=="Unidentified")return e}return t.type==="keypress"?(t=gu(t),t===13?"Enter":String.fromCharCode(t)):t.type==="keydown"||t.type==="keyup"?$E[t.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Pm,charCode:function(t){return t.type==="keypress"?gu(t):0},keyCode:function(t){return t.type==="keydown"||t.type==="keyup"?t.keyCode:0},which:function(t){return t.type==="keypress"?gu(t):t.type==="keydown"||t.type==="keyup"?t.keyCode:0}}),iT=Pn(nT),sT=xt({},ff,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),lx=Pn(sT),rT=xt({},Er,{submitter:0}),aT=Pn(rT),oT=xt({},jl,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Pm}),lT=Pn(oT),cT=xt({},Er,{propertyName:0,elapsedTime:0,pseudoElement:0}),uT=Pn(cT),fT=xt({},ff,{deltaX:function(t){return"deltaX"in t?t.deltaX:"wheelDeltaX"in t?-t.wheelDeltaX:0},deltaY:function(t){return"deltaY"in t?t.deltaY:"wheelDeltaY"in t?-t.wheelDeltaY:"wheelDelta"in t?-t.wheelDelta:0},deltaZ:0,deltaMode:0}),hT=Pn(fT),dT=xt({},Er,{newState:0,oldState:0,source:0}),pT=Pn(dT),mT=[9,13,27,32],Om=Cs&&"CompositionEvent"in window,xl=null;Cs&&"documentMode"in document&&(xl=document.documentMode);var gT=Cs&&"TextEvent"in window&&!xl,s_=Cs&&(!Om||xl&&8<xl&&11>=xl),cx=" ",ux=!1;function r_(t,e){switch(t){case"keyup":return mT.indexOf(e.keyCode)!==-1;case"keydown":return e.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function a_(t){return t=t.detail,typeof t=="object"&&"data"in t?t.data:null}var Va=!1;function vT(t,e){switch(t){case"compositionend":return a_(e);case"keypress":return e.which!==32?null:(ux=!0,cx);case"textInput":return t=e.data,t===cx&&ux?null:t;default:return null}}function xT(t,e){if(Va)return t==="compositionend"||!Om&&r_(t,e)?(t=i_(),mu=Lm=sr=null,Va=!1,t):null;switch(t){case"paste":return null;case"keypress":if(!(e.ctrlKey||e.altKey||e.metaKey)||e.ctrlKey&&e.altKey){if(e.char&&1<e.char.length)return e.char;if(e.which)return String.fromCharCode(e.which)}return null;case"compositionend":return s_&&e.locale!=="ko"?null:e.data;default:return null}}var yT={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function fx(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e==="input"?!!yT[t.type]:e==="textarea"}function o_(t,e,n,i){Ga?ja?ja.push(i):ja=[i]:Ga=i,e=rf(e,"onChange"),0<e.length&&(n=new uf("onChange","change",null,n,i),t.push({event:n,listeners:e}))}var yl=null,Bl=null;function _T(t){nA(t,0)}function hf(t){var e=ml(t);if(Jy(e))return t}function hx(t,e){if(t==="change")return e}var l_=!1;Cs&&(Cs?($c="oninput"in document,$c||(Kd=document.createElement("div"),Kd.setAttribute("oninput","return;"),$c=typeof Kd.oninput=="function"),jc=$c):jc=!1,l_=jc&&(!document.documentMode||9<document.documentMode));var jc,$c,Kd;function dx(){yl&&(yl.detachEvent("onpropertychange",c_),Bl=yl=null)}function c_(t){if(t.propertyName==="value"&&hf(Bl)){var e=[];o_(e,Bl,t,Nm(t)),n_(_T,e)}}function ST(t,e,n){t==="focusin"?(dx(),yl=e,Bl=n,yl.attachEvent("onpropertychange",c_)):t==="focusout"&&dx()}function AT(t){if(t==="selectionchange"||t==="keyup"||t==="keydown")return hf(Bl)}function MT(t,e){if(t==="click")return hf(e)}function ET(t,e){if(t==="input"||t==="change")return hf(e)}function TT(t,e){return t===e&&(t!==0||1/t===1/e)||t!==t&&e!==e}var si=typeof Object.is=="function"?Object.is:TT;function Nl(t,e){if(si(t,e))return!0;if(typeof t!="object"||t===null||typeof e!="object"||e===null)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(i=0;i<n.length;i++){var s=n[i];if(!wp.call(e,s)||!si(t[s],e[s]))return!1}return!0}function Ip(t){if(t=t||(typeof document<"u"?document:void 0),typeof t>"u")return null;try{return t.activeElement||t.body}catch{return t.body}}function px(t){for(;t&&t.firstChild;)t=t.firstChild;return t}function mx(t,e){var n=px(t);t=0;for(var i;n;){if(n.nodeType===3){if(i=t+n.textContent.length,t<=e&&i>=e)return{node:n,offset:e-t};t=i}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=px(n)}}function u_(t,e){return t&&e?t===e?!0:t&&t.nodeType===3?!1:e&&e.nodeType===3?u_(t,e.parentNode):"contains"in t?t.contains(e):t.compareDocumentPosition?!!(t.compareDocumentPosition(e)&16):!1:!1}function f_(t){t=t!=null&&t.ownerDocument!=null&&t.ownerDocument.defaultView!=null?t.ownerDocument.defaultView:window;for(var e=Ip(t.document);e instanceof t.HTMLIFrameElement;){try{var n=typeof e.contentWindow.location.href=="string"}catch{n=!1}if(n)t=e.contentWindow;else break;e=Ip(t.document)}return e}function Fm(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&(e==="input"&&(t.type==="text"||t.type==="search"||t.type==="tel"||t.type==="url"||t.type==="password")||e==="textarea"||t.contentEditable==="true")}var bT=Cs&&"documentMode"in document&&11>=document.documentMode,ka=null,Bp=null,_l=null,Np=!1;function gx(t,e,n){var i=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Np||ka==null||ka!==Ip(i)||(i=ka,"selectionStart"in i&&Fm(i)?i={start:i.selectionStart,end:i.selectionEnd}:(i=(i.ownerDocument&&i.ownerDocument.defaultView||window).getSelection(),i={anchorNode:i.anchorNode,anchorOffset:i.anchorOffset,focusNode:i.focusNode,focusOffset:i.focusOffset}),_l&&Nl(_l,i)||(_l=i,i=rf(Bp,"onSelect"),0<i.length&&(e=new uf("onSelect","select",null,e,n),t.push({event:e,listeners:i}),e.target=ka)))}function Xr(t,e){var n={};return n[t.toLowerCase()]=e.toLowerCase(),n["Webkit"+t]="webkit"+e,n["Moz"+t]="moz"+e,n}var Wa={animationend:Xr("Animation","AnimationEnd"),animationiteration:Xr("Animation","AnimationIteration"),animationstart:Xr("Animation","AnimationStart"),transitionrun:Xr("Transition","TransitionRun"),transitionstart:Xr("Transition","TransitionStart"),transitioncancel:Xr("Transition","TransitionCancel"),transitionend:Xr("Transition","TransitionEnd")},Jd={},h_={};Cs&&(h_=document.createElement("div").style,"AnimationEvent"in window||(delete Wa.animationend.animation,delete Wa.animationiteration.animation,delete Wa.animationstart.animation),"TransitionEvent"in window||delete Wa.transitionend.transition);function da(t){if(Jd[t])return Jd[t];if(!Wa[t])return t;var e=Wa[t],n;for(n in e)if(e.hasOwnProperty(n)&&n in h_)return Jd[t]=e[n];return t}var d_=da("animationend"),p_=da("animationiteration"),m_=da("animationstart"),wT=da("transitionrun"),CT=da("transitionstart"),RT=da("transitioncancel"),g_=da("transitionend"),v_=new Map,Lp="abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");Lp.push("scrollEnd");function Oi(t,e){v_.set(t,e),ha(e,[t])}var DT=0;function Rs(t,e){if(t.name!=null&&t.name!=="auto")return t.name;if(e.autoName!==null)return e.autoName;t=Pi.identifierPrefix;var n=DT++;return t="_"+t+"t_"+n.toString(32)+"_",e.autoName=t}function vx(t){if(t==null||typeof t=="string")return t;var e=null,n=ro;if(n!==null)for(var i=0;i<n.length;i++){var s=t[n[i]];if(s!=null){if(s==="none")return"none";e=e==null?s:e+(" "+s)}}return e??t.default}function Ls(t,e){return t=vx(t),e=vx(e),e==null?t==="auto"?null:t:e==="auto"?null:e}var Ou=typeof reportError=="function"?reportError:function(t){if(typeof window=="object"&&typeof window.ErrorEvent=="function"){var e=new window.ErrorEvent("error",{bubbles:!0,cancelable:!0,message:typeof t=="object"&&t!==null&&typeof t.message=="string"?String(t.message):String(t),error:t});if(!window.dispatchEvent(e))return}else if(typeof process=="object"&&typeof process.emit=="function"){process.emit("uncaughtException",t);return}console.error(t)},yi=[],Xa=0,zm=0;function df(){for(var t=Xa,e=zm=Xa=0;e<t;){var n=yi[e];yi[e++]=null;var i=yi[e];yi[e++]=null;var s=yi[e];yi[e++]=null;var r=yi[e];if(yi[e++]=null,i!==null&&s!==null){var a=i.pending;a===null?s.next=s:(s.next=a.next,a.next=s),i.pending=s}r!==0&&x_(n,s,r)}}function pf(t,e,n,i){yi[Xa++]=t,yi[Xa++]=e,yi[Xa++]=n,yi[Xa++]=i,zm|=i,t.lanes|=i,t=t.alternate,t!==null&&(t.lanes|=i)}function Hm(t,e,n,i){return pf(t,e,n,i),Fu(t)}function pa(t,e){return pf(t,null,null,e),Fu(t)}function x_(t,e,n){t.lanes|=n;var i=t.alternate;i!==null&&(i.lanes|=n);for(var s=!1,r=t.return;r!==null;)r.childLanes|=n,i=r.alternate,i!==null&&(i.childLanes|=n),r.tag===22&&(t=r.stateNode,t===null||t._visibility&1||(s=!0)),t=r,r=r.return;return t.tag===3?(r=t.stateNode,s&&e!==null&&(s=31-ni(n),t=r.hiddenUpdates,i=t[s],i===null?t[s]=[e]:i.push(e),e.lane=n|536870912),r):null}function Fu(t){if(50<Dl)throw Dl=0,bu=null,Error(J(185));for(var e=t.return;e!==null;)t=e,e=t.return;return t.tag===3?t.stateNode:null}var Ya={};function UT(t,e,n,i){this.tag=t,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=e,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=i,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function kn(t,e,n,i){return new UT(t,e,n,i)}function Gm(t){return t=t.prototype,!(!t||!t.isReactComponent)}function bs(t,e){var n=t.alternate;return n===null?(n=kn(t.tag,e,t.key,t.mode),n.elementType=t.elementType,n.type=t.type,n.stateNode=t.stateNode,n.alternate=t,t.alternate=n):(n.pendingProps=e,n.type=t.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=t.flags&1206910976,n.childLanes=t.childLanes,n.lanes=t.lanes,n.child=t.child,n.memoizedProps=t.memoizedProps,n.memoizedState=t.memoizedState,n.updateQueue=t.updateQueue,e=t.dependencies,n.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext},n.sibling=t.sibling,n.index=t.index,n.ref=t.ref,n.refCleanup=t.refCleanup,n}function y_(t,e){t.flags&=1206910978;var n=t.alternate;return n===null?(t.childLanes=0,t.lanes=e,t.child=null,t.subtreeFlags=0,t.memoizedProps=null,t.memoizedState=null,t.updateQueue=null,t.dependencies=null,t.stateNode=null):(t.childLanes=n.childLanes,t.lanes=n.lanes,t.child=n.child,t.subtreeFlags=0,t.deletions=null,t.memoizedProps=n.memoizedProps,t.memoizedState=n.memoizedState,t.updateQueue=n.updateQueue,t.type=n.type,e=n.dependencies,t.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),t}function vu(t,e,n,i,s,r){var a=0;if(i=t,typeof i=="function")Gm(i)&&(a=1);else if(typeof i=="string")a=iw(t,n,as.current)?26:t==="html"||t==="head"||t==="body"?27:5;else e:switch(i){case Ap:return t=kn(31,n,e,s),t.elementType=Ap,t.lanes=r,t;case za:return $r(n.children,s,r,e);case Py:a=8,s|=24;break;case yp:return t=kn(12,n,e,s|2),t.elementType=yp,t.lanes=r,t;case _p:return t=kn(13,n,e,s),t.elementType=_p,t.lanes=r,t;case Sp:return t=kn(19,n,e,s),t.elementType=Sp,t.lanes=r,t;case xE:case Mp:return t=s|32,t=kn(30,n,e,t),t.elementType=Mp,t.lanes=r,t.stateNode={autoName:null,paired:null,clones:null,ref:null},t;default:if(typeof i=="object"&&i!==null)switch(i.$$typeof){case ns:a=10;break e;case Oy:a=9;break e;case Cm:a=11;break e;case Rm:a=14;break e;case er:a=16,i=null;break e}a=29,n=Error(J(130,t===null?"null":typeof t,"")),i=null}return e=kn(a,n,e,s),e.elementType=t,e.type=i,e.lanes=r,e}function $r(t,e,n,i){return t=kn(7,t,i,e),t.lanes=n,t}function jd(t,e,n){return t=kn(6,t,null,e),t.lanes=n,t}function __(t){var e=kn(18,null,null,0);return e.stateNode=t,e}function $d(t,e,n){return e=kn(4,t.children!==null?t.children:[],t.key,e),e.lanes=n,e.stateNode={containerInfo:t.containerInfo,pendingChildren:null,implementation:t.implementation},e}var xx=new WeakMap;function Mi(t,e){if(typeof t=="object"&&t!==null){var n=xx.get(t);return n!==void 0?n:(e={value:t,source:e,stack:K0(e)},xx.set(t,e),e)}return{value:t,source:e,stack:K0(e)}}var qa=[],Qa=0,zu=null,Ll=0,_i=[],Si=0,yr=null,ss=1,rs="";function Es(t,e){qa[Qa++]=Ll,qa[Qa++]=zu,zu=t,Ll=e}function S_(t,e,n){_i[Si++]=ss,_i[Si++]=rs,_i[Si++]=yr,yr=t;var i=ss;t=rs;var s=32-ni(i)-1;i&=~(1<<s),n+=1;var r=32-ni(e)+s;if(30<r){var a=s-s%5;r=(i&(1<<a)-1).toString(32),i>>=a,s-=a,ss=1<<32-ni(e)+s|n<<s|i,rs=r+t}else ss=1<<r|n<<s|i,rs=t}function mf(t){t.return!==null&&(Es(t,1),S_(t,1,0))}function Vm(t){for(;t===zu;)zu=qa[--Qa],qa[Qa]=null,Ll=qa[--Qa],qa[Qa]=null;for(;t===yr;)yr=_i[--Si],_i[Si]=null,rs=_i[--Si],_i[Si]=null,ss=_i[--Si],_i[Si]=null}function A_(t,e){_i[Si++]=ss,_i[Si++]=rs,_i[Si++]=yr,ss=e.id,rs=e.overflow,yr=t}var cn=null,At=null,ke=!1,ur=null,Ei=!1,Pp=Error(J(519));function _r(t){var e=Error(J(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?"text":"HTML",""));throw Pl(Mi(e,t)),Pp}function yx(t){var e=t.stateNode,n=t.type,i=t.memoizedProps;switch(e[gn]=t,e[Yn]=i,n){case"dialog":We("cancel",e),We("close",e);break;case"iframe":case"object":case"embed":We("load",e);break;case"video":case"audio":for(n=0;n<Hl.length;n++)We(Hl[n],e);break;case"source":We("error",e);break;case"img":case"image":case"link":We("error",e),We("load",e);break;case"details":We("toggle",e);break;case"input":We("invalid",e),jy(e,i.value,i.defaultValue,i.checked,i.defaultChecked,i.type,i.name,!0);break;case"select":We("invalid",e);break;case"textarea":We("invalid",e),e_(e,i.value,i.defaultValue,i.children)}n=i.children,typeof n!="string"&&typeof n!="number"&&typeof n!="bigint"||e.textContent===""+n||i.suppressHydrationWarning===!0||sA(e.textContent,n)?(i.popover!=null&&(We("beforetoggle",e),We("toggle",e)),i.onScroll!=null&&We("scroll",e),i.onScrollEnd!=null&&We("scrollend",e),i.onClick!=null&&(e.onclick=is),e=!0):e=!1,e||_r(t,!0)}function Hu(t){for(cn=t.return;cn;)switch(cn.tag){case 5:case 31:case 13:Ei=!1;return;case 27:case 3:Ei=!0;return;default:cn=cn.return}}function Ba(t){if(t!==cn)return!1;if(!ke)return Hu(t),ke=!0,!1;var e=t.tag,n;if((n=e!==3&&e!==27)&&((n=e===5)&&(n=t.type,n=!(n!=="form"&&n!=="button")||_m(t.type,t.memoizedProps)),n=!n),n&&At&&_r(t),Hu(t),e===13){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));At=my(t)}else if(e===31){if(t=t.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(317));At=my(t)}else e===27?(e=At,Tr(t.type)?(t=Em,Em=null,At=t):At=e):At=cn?Ti(t.stateNode.nextSibling):null;return!0}function ia(){At=cn=null,ke=!1}function ep(){var t=ur;return t!==null&&(Gn===null?Gn=t:Gn.push.apply(Gn,t),ur=null),t}function Pl(t){ur===null?ur=[t]:ur.push(t)}var Op=cs(null),ma=null,Ts=null;function rr(t,e,n){Mt(Op,e._currentValue),e._currentValue=n}function ws(t){t._currentValue=Op.current,xn(Op)}function xu(t,e,n){for(;t!==null;){var i=t.alternate;if((t.childLanes&e)!==e?(t.childLanes|=e,i!==null&&(i.childLanes|=e)):i!==null&&(i.childLanes&e)!==e&&(i.childLanes|=e),t===n)break;t=t.return}}function Fp(t,e,n,i){var s=t.child;for(s!==null&&(s.return=t);s!==null;){var r=s.dependencies;if(r!==null){var a=s.child;r=r.firstContext;e:for(;r!==null;){var o=r;r=s;for(var l=0;l<e.length;l++)if(o.context===e[l]){r.lanes|=n,o=r.alternate,o!==null&&(o.lanes|=n),xu(r.return,n,t),i||(a=null);break e}r=o.next}}else if(s.tag===18){if(a=s.return,a===null)throw Error(J(341));a.lanes|=n,r=a.alternate,r!==null&&(r.lanes|=n),xu(a,n,t),a=null}else s.tag===13&&s.memoizedState!==null&&s.memoizedState.dehydrated===null?(s.lanes|=n,a=s.alternate,a!==null&&(a.lanes|=n),xu(s.return,n,t),a=s.child,a=a!==null?a.sibling:null):a=s.child;if(a!==null)a.return=s;else for(a=s;a!==null;){if(a===t){a=null;break}if(s=a.sibling,s!==null){s.return=a.return,a=s;break}a=a.return}s=a}}function sa(t,e,n,i){t=null;for(var s=e,r=!1;s!==null;){if(!r){if((s.flags&524288)!==0)r=!0;else if((s.flags&262144)!==0)break}if(s.tag===10){var a=s.alternate;if(a===null)throw Error(J(387));if(a=a.memoizedProps,a!==null){var o=s.type;si(s.pendingProps.value,a.value)||(t!==null?t.push(o):t=[o])}}else if(s===Iu.current){if(a=s.alternate,a===null)throw Error(J(387));a.memoizedState.memoizedState!==s.memoizedState.memoizedState&&(t!==null?t.push(vo):t=[vo])}s=s.return}return t!==null&&Fp(e,t,n,i),e.flags|=262144,t!==null}function Gu(t){for(t=t.firstContext;t!==null;){if(!si(t.context._currentValue,t.memoizedValue))return!0;t=t.next}return!1}function ra(t){ma=t,Ts=null,t=t.dependencies,t!==null&&(t.firstContext=null)}function vn(t){return M_(ma,t)}function eu(t,e){return ma===null&&ra(t),M_(t,e)}function M_(t,e){var n=e._currentValue;if(e={context:e,memoizedValue:n,next:null},Ts===null){if(t===null)throw Error(J(308));Ts=e,t.dependencies={lanes:0,firstContext:e},t.flags|=524288}else Ts=Ts.next=e;return n}var IT=typeof AbortController<"u"?AbortController:function(){var t=[],e=this.signal={aborted:!1,addEventListener:function(n,i){t.push(i)}};this.abort=function(){e.aborted=!0,t.forEach(function(n){return n()})}},BT=$t.unstable_scheduleCallback,NT=$t.unstable_NormalPriority,Wt={$$typeof:ns,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function km(){return{controller:new IT,data:new Map,refCount:0}}function $l(t){t.refCount--,t.refCount===0&&BT(NT,function(){t.controller.abort()})}function _x(t,e){if((t.pendingLanes&4194048)!==0){var n=t.transitionTypes;for(n===null&&(n=t.transitionTypes=[]),t=0;t<e.length;t++){var i=e[t];n.indexOf(i)===-1&&n.push(i)}}}var gl=null;function LT(t){var e=t.transitionTypes;return t.transitionTypes=null,e}var Sl=null,zp=0,aa=0,$a=null;function PT(t,e){if(Sl===null){var n=Sl=[];zp=0,aa=vg(),$a={status:"pending",value:void 0,then:function(i){n.push(i)}}}return zp++,e.then(Sx,Sx),e}function Sx(){if(--zp===0&&(gl=null,Sl!==null)){$a!==null&&($a.status="fulfilled");var t=Sl;Sl=null,aa=0,$a=null;for(var e=0;e<t.length;e++)(0,t[e])()}}function OT(t,e){var n=[],i={status:"pending",value:null,reason:null,then:function(s){n.push(s)}};return t.then(function(){i.status="fulfilled",i.value=e;for(var s=0;s<n.length;s++)(0,n[s])(e)},function(s){for(i.status="rejected",i.reason=s,s=0;s<n.length;s++)(0,n[s])(void 0)}),i}var Ax=Ue.S;Ue.S=function(t,e){if(GS=ei(),typeof e=="object"&&e!==null&&typeof e.then=="function"&&PT(t,e),gl!==null)for(var n=po;n!==null;)_x(n,gl),n=n.next;if(n=t.types,n!==null){for(var i=po;i!==null;)_x(i,n),i=i.next;if(aa!==0){i=gl,i===null&&(i=gl=[]);for(var s=0;s<n.length;s++){var r=n[s];i.indexOf(r)===-1&&i.push(r)}}}Ax!==null&&Ax(t,e)};var ea=cs(null);function Wm(){var t=ea.current;return t!==null?t:vt.pooledCache}function yu(t,e){e===null?Mt(ea,ea.current):Mt(ea,e.pool)}function E_(){var t=Wm();return t===null?null:{parent:Wt._currentValue,pool:t}}var Ao=Error(J(460)),Xm=Error(J(474)),gf=Error(J(542)),Vu={then:function(){}};function Mx(t){return t=t.status,t==="fulfilled"||t==="rejected"}function T_(t,e,n){switch(n=t[n],n===void 0?t.push(e):n!==e&&(e.then(is,is),e=n),e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,Tx(t),t===void 0&&!("reason"in e)?Error(J(600)):t;default:if(typeof e.status=="string")e.then(is,is);else{if(t=vt,t!==null&&100<t.shellSuspendCounter)throw Error(J(482));t=e,t.status="pending",t.then(function(i){if(e.status==="pending"){var s=e;s.status="fulfilled",s.value=i}},function(i){if(e.status==="pending"){var s=e;s.status="rejected",s.reason=i}})}switch(e.status){case"fulfilled":return e.value;case"rejected":throw t=e.reason,Tx(t),t}throw ta=e,Ao}}function Qr(t){try{var e=t._init;return e(t._payload)}catch(n){throw n!==null&&typeof n=="object"&&typeof n.then=="function"?(ta=n,Ao):n}}var ta=null;function Ex(){if(ta===null)throw Error(J(459));var t=ta;return ta=null,t}function Tx(t){if(t===Ao||t===gf)throw Error(J(483))}var eo=null,Ol=0;function tu(t){var e=Ol;return Ol+=1,eo===null&&(eo=[]),T_(eo,t,e)}function Js(t,e){e=e.props.ref,t.ref=e!==void 0?e:null}function nu(t,e){throw e.$$typeof===vE?Error(J(525)):(t=Object.prototype.toString.call(e),Error(J(31,t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)))}function b_(t){function e(f,m){if(t){var y=f.deletions;y===null?(f.deletions=[m],f.flags|=16):y.push(m)}}function n(f,m){if(!t)return null;for(;m!==null;)e(f,m),m=m.sibling;return null}function i(f){for(var m=new Map;f!==null;)f.key===null?m.set(f.index,f):m.set(f.key,f),f=f.sibling;return m}function s(f,m){return f=bs(f,m),f.index=0,f.sibling=null,f}function r(f,m,y){return f.index=y,t?(y=f.alternate,y!==null?(y=y.index,y<m?(f.flags|=2,m):y):(f.flags|=134217730,m)):(f.flags|=1048576,m)}function a(f){return t&&f.alternate===null&&(f.flags|=134217730),f}function o(f,m,y,_){return m===null||m.tag!==6?(m=jd(y,f.mode,_),m.return=f,m):(m=s(m,y),m.return=f,m)}function l(f,m,y,_){var T=y.type;return T===za?(f=h(f,m,y.props.children,_,y.key),Js(f,y),f):m!==null&&(m.elementType===T||typeof T=="object"&&T!==null&&T.$$typeof===er&&Qr(T)===m.type)?(m=s(m,y.props),Js(m,y),m.return=f,m):(m=vu(y.type,y.key,y.props,null,f.mode,_),Js(m,y),m.return=f,m)}function c(f,m,y,_){return m===null||m.tag!==4||m.stateNode.containerInfo!==y.containerInfo||m.stateNode.implementation!==y.implementation?(m=$d(y,f.mode,_),m.return=f,m):(m=s(m,y.children||[]),m.return=f,m)}function h(f,m,y,_,T){return m===null||m.tag!==7?(m=$r(y,f.mode,_,T),m.return=f,m):(m=s(m,y),m.return=f,m)}function p(f,m,y){if(typeof m=="string"&&m!==""||typeof m=="number"||typeof m=="bigint")return m=jd(""+m,f.mode,y),m.return=f,m;if(typeof m=="object"&&m!==null){switch(m.$$typeof){case Yc:return y=vu(m.type,m.key,m.props,null,f.mode,y),Js(y,m),y.return=f,y;case dl:return m=$d(m,f.mode,y),m.return=f,m;case er:return m=Qr(m),p(f,m,y)}if(pl(m)||ol(m))return m=$r(m,f.mode,y,null),m.return=f,m;if(typeof m.then=="function")return p(f,tu(m),y);if(m.$$typeof===ns)return p(f,eu(f,m),y);nu(f,m)}return null}function u(f,m,y,_){var T=m!==null?m.key:null;if(typeof y=="string"&&y!==""||typeof y=="number"||typeof y=="bigint")return T!==null?null:o(f,m,""+y,_);if(typeof y=="object"&&y!==null){switch(y.$$typeof){case Yc:return y.key===T?l(f,m,y,_):null;case dl:return y.key===T?c(f,m,y,_):null;case er:return y=Qr(y),u(f,m,y,_)}if(pl(y)||ol(y))return T!==null?null:h(f,m,y,_,null);if(typeof y.then=="function")return u(f,m,tu(y),_);if(y.$$typeof===ns)return u(f,m,eu(f,y),_);nu(f,y)}return null}function d(f,m,y,_,T){if(typeof _=="string"&&_!==""||typeof _=="number"||typeof _=="bigint")return f=f.get(y)||null,o(m,f,""+_,T);if(typeof _=="object"&&_!==null){switch(_.$$typeof){case Yc:return f=f.get(_.key===null?y:_.key)||null,l(m,f,_,T);case dl:return f=f.get(_.key===null?y:_.key)||null,c(m,f,_,T);case er:return _=Qr(_),d(f,m,y,_,T)}if(pl(_)||ol(_))return f=f.get(y)||null,h(m,f,_,T,null);if(typeof _.then=="function")return d(f,m,y,tu(_),T);if(_.$$typeof===ns)return d(f,m,y,eu(m,_),T);nu(m,_)}return null}function x(f,m,y,_){for(var T=null,b=null,v=m,E=m=0,N=null;v!==null&&E<y.length;E++){v.index>E?(N=v,v=null):N=v.sibling;var C=u(f,v,y[E],_);if(C===null){v===null&&(v=N);break}t&&v&&C.alternate===null&&e(f,v),m=r(C,m,E),b===null?T=C:b.sibling=C,b=C,v=N}if(E===y.length)return n(f,v),ke&&Es(f,E),T;if(v===null){for(;E<y.length;E++)v=p(f,y[E],_),v!==null&&(m=r(v,m,E),b===null?T=v:b.sibling=v,b=v);return ke&&Es(f,E),T}for(v=i(v);E<y.length;E++)N=d(v,f,E,y[E],_),N!==null&&(t&&(C=N.alternate,C!==null&&v.delete(C.key===null?E:C.key)),m=r(N,m,E),b===null?T=N:b.sibling=N,b=N);return t&&v.forEach(function(B){return e(f,B)}),ke&&Es(f,E),T}function A(f,m,y,_){if(y==null)throw Error(J(151));for(var T=null,b=null,v=m,E=m=0,N=null,C=y.next();v!==null&&!C.done;E++,C=y.next()){v.index>E?(N=v,v=null):N=v.sibling;var B=u(f,v,C.value,_);if(B===null){v===null&&(v=N);break}t&&v&&B.alternate===null&&e(f,v),m=r(B,m,E),b===null?T=B:b.sibling=B,b=B,v=N}if(C.done)return n(f,v),ke&&Es(f,E),T;if(v===null){for(;!C.done;E++,C=y.next())C=p(f,C.value,_),C!==null&&(m=r(C,m,E),b===null?T=C:b.sibling=C,b=C);return ke&&Es(f,E),T}for(v=i(v);!C.done;E++,C=y.next())C=d(v,f,E,C.value,_),C!==null&&(t&&(N=C.alternate,N!==null&&v.delete(N.key===null?E:N.key)),m=r(C,m,E),b===null?T=C:b.sibling=C,b=C);return t&&v.forEach(function(F){return e(f,F)}),ke&&Es(f,E),T}function g(f,m,y,_){if(typeof y=="object"&&y!==null&&y.type===za&&y.key===null&&y.props.ref===void 0&&(y=y.props.children),typeof y=="object"&&y!==null){switch(y.$$typeof){case Yc:e:{for(var T=y.key;m!==null;){if(m.key===T){if(T=y.type,T===za){if(m.tag===7){n(f,m.sibling),_=s(m,y.props.children),Js(_,y),_.return=f,f=_;break e}}else if(m.elementType===T||typeof T=="object"&&T!==null&&T.$$typeof===er&&Qr(T)===m.type){n(f,m.sibling),_=s(m,y.props),Js(_,y),_.return=f,f=_;break e}n(f,m);break}else e(f,m);m=m.sibling}y.type===za?(_=$r(y.props.children,f.mode,_,y.key),Js(_,y),_.return=f,f=_):(_=vu(y.type,y.key,y.props,null,f.mode,_),Js(_,y),_.return=f,f=_)}return a(f);case dl:e:{for(T=y.key;m!==null;){if(m.key===T)if(m.tag===4&&m.stateNode.containerInfo===y.containerInfo&&m.stateNode.implementation===y.implementation){n(f,m.sibling),_=s(m,y.children||[]),_.return=f,f=_;break e}else{n(f,m);break}else e(f,m);m=m.sibling}_=$d(y,f.mode,_),_.return=f,f=_}return a(f);case er:return y=Qr(y),g(f,m,y,_)}if(pl(y))return x(f,m,y,_);if(ol(y)){if(T=ol(y),typeof T!="function")throw Error(J(150));return y=T.call(y),A(f,m,y,_)}if(typeof y.then=="function")return g(f,m,tu(y),_);if(y.$$typeof===ns)return g(f,m,eu(f,y),_);nu(f,y)}return typeof y=="string"&&y!==""||typeof y=="number"||typeof y=="bigint"?(y=""+y,m!==null&&m.tag===6?(n(f,m.sibling),_=s(m,y),_.return=f,f=_):(n(f,m),_=jd(y,f.mode,_),_.return=f,f=_),a(f)):n(f,m)}return function(f,m,y,_){try{Ol=0;var T=g(f,m,y,_);return eo=null,T}catch(v){if(v===Ao||v===gf)throw v;var b=kn(29,v,null,f.mode);return b.lanes=_,b.return=f,b}}}var oa=b_(!0),w_=b_(!1),tr=!1;function Ym(t){t.updateQueue={baseState:t.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function Hp(t,e){t=t.updateQueue,e.updateQueue===t&&(e.updateQueue={baseState:t.baseState,firstBaseUpdate:t.firstBaseUpdate,lastBaseUpdate:t.lastBaseUpdate,shared:t.shared,callbacks:null})}function fr(t){return{lane:t,tag:0,payload:null,callback:null,next:null}}function hr(t,e,n){var i=t.updateQueue;if(i===null)return null;if(i=i.shared,(at&2)!==0){var s=i.pending;return s===null?e.next=e:(e.next=s.next,s.next=e),i.pending=e,e=Fu(t),x_(t,null,n),e}return pf(t,i,e,n),Fu(t)}function Al(t,e,n){if(e=e.updateQueue,e!==null&&(e=e.shared,(n&4194048)!==0)){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,Wy(t,n)}}function tp(t,e){var n=t.updateQueue,i=t.alternate;if(i!==null&&(i=i.updateQueue,n===i)){var s=null,r=null;if(n=n.firstBaseUpdate,n!==null){do{var a={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};r===null?s=r=a:r=r.next=a,n=n.next}while(n!==null);r===null?s=r=e:r=r.next=e}else s=r=e;n={baseState:i.baseState,firstBaseUpdate:s,lastBaseUpdate:r,shared:i.shared,callbacks:i.callbacks},t.updateQueue=n;return}t=n.lastBaseUpdate,t===null?n.firstBaseUpdate=e:t.next=e,n.lastBaseUpdate=e}var Gp=!1;function Ml(){if(Gp){var t=$a;if(t!==null)throw t}}function El(t,e,n,i){Gp=!1;var s=t.updateQueue;tr=!1;var r=s.firstBaseUpdate,a=s.lastBaseUpdate,o=s.shared.pending;if(o!==null){s.shared.pending=null;var l=o,c=l.next;l.next=null,a===null?r=c:a.next=c,a=l;var h=t.alternate;h!==null&&(h=h.updateQueue,o=h.lastBaseUpdate,o!==a&&(o===null?h.firstBaseUpdate=c:o.next=c,h.lastBaseUpdate=l))}if(r!==null){var p=s.baseState;a=0,h=c=l=null,o=r;do{var u=o.lane&-536870913,d=u!==o.lane;if(d?(Je&u)===u:(i&u)===u){u!==0&&u===aa&&(Gp=!0),h!==null&&(h=h.next={lane:0,tag:o.tag,payload:o.payload,callback:null,next:null});e:{var x=t,A=o;u=e;var g=n;switch(A.tag){case 1:if(x=A.payload,typeof x=="function"){p=x.call(g,p,u);break e}p=x;break e;case 3:x.flags=x.flags&-65537|128;case 0:if(x=A.payload,u=typeof x=="function"?x.call(g,p,u):x,u==null)break e;p=xt({},p,u);break e;case 2:tr=!0}}u=o.callback,u!==null&&(t.flags|=64,d&&(t.flags|=8192),d=s.callbacks,d===null?s.callbacks=[u]:d.push(u))}else d={lane:u,tag:o.tag,payload:o.payload,callback:o.callback,next:null},h===null?(c=h=d,l=p):h=h.next=d,a|=u;if(o=o.next,o===null){if(o=s.shared.pending,o===null)break;d=o,o=d.next,d.next=null,s.lastBaseUpdate=d,s.shared.pending=null}}while(!0);h===null&&(l=p),s.baseState=l,s.firstBaseUpdate=c,s.lastBaseUpdate=h,r===null&&(s.shared.lanes=0),Mr|=a,t.lanes=a,t.memoizedState=p}}function C_(t,e){if(typeof t!="function")throw Error(J(191,t));t.call(e)}function R_(t,e){var n=t.callbacks;if(n!==null)for(t.callbacks=null,t=0;t<n.length;t++)C_(n[t],e)}var Sr=cs(null),ku=cs(0);function bx(t,e){t=Bs,Mt(ku,t),Mt(Sr,e),Bs=t|e.baseLanes}function Vp(){Mt(ku,Bs),Mt(Sr,Sr.current)}function qm(){Bs=ku.current,xn(Sr),xn(ku)}var Sn=cs(null),bn=null;function dr(t){var e=t.alternate;Mt(yn,yn.current&1),Mt(Sn,t),bn===null&&(e===null||Sr.current!==null||e.memoizedState!==null)&&(bn=t)}function kp(t){Mt(yn,yn.current),Mt(Sn,t),bn===null&&(bn=t)}function D_(t){t.tag===22?(Mt(yn,yn.current),Mt(Sn,t),bn===null&&(bn=t)):pr()}function pr(){Mt(yn,yn.current),Mt(Sn,Sn.current)}function Jn(t){xn(Sn),bn===t&&(bn=null),xn(yn)}var yn=cs(0);function Fl(t,e){Mt(Sn,Sn.current),Mt(yn,e)}function Qm(t){xn(yn),xn(Sn),bn===t&&(bn=null)}function Wu(t){for(var e=t;e!==null;){if(e.tag===13){var n=e.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||Mm(n)||Sg(n)))return e}else if(e.tag===19&&e.memoizedProps.revealOrder!=="independent"){if((e.flags&128)!==0)return e}else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return null;e=e.return}e.sibling.return=e.return,e=e.sibling}return null}var Ds=0,ze=null,pt=null,kt=null,Xu=!1,to=!1,la=!1,Yu=0,zl=0,no=null,FT=0;function Pt(){throw Error(J(321))}function Zm(t,e){if(e===null)return!1;for(var n=0;n<e.length&&n<t.length;n++)if(!si(t[n],e[n]))return!1;return!0}function Km(t,e,n,i,s,r){return Ds=r,ze=e,e.memoizedState=null,e.updateQueue=null,e.lanes=0,Ue.H=t===null||t.memoizedState===null?oS:lS,la=!1,r=n(i,s),la=!1,to&&(r=I_(e,n,i,s)),U_(t),r}function U_(t){Ue.H=qu;var e=pt!==null&&pt.next!==null;if(Ds=0,kt=pt=ze=null,Xu=!1,zl=0,no=null,e)throw Error(J(300));t===null||Xt||(t=t.dependencies,t!==null&&Gu(t)&&(Xt=!0))}function I_(t,e,n,i){ze=t;var s=0;do{if(to&&(no=null),zl=0,to=!1,25<=s)throw Error(J(301));if(s+=1,kt=pt=null,t.updateQueue!=null){var r=t.updateQueue;r.lastEffect=null,r.events=null,r.stores=null,r.memoCache!=null&&(r.memoCache.index=0)}Ue.H=YT,r=e(n,i)}while(to);return r}function zT(){var t=Ue.H,e=t.useState()[0];return e=typeof e.then=="function"?ec(e):e,t=t.useState()[0],(pt!==null?pt.memoizedState:null)!==t&&(ze.flags|=1024),e}function Jm(){var t=Yu!==0;return Yu=0,t}function jm(t,e,n){e.updateQueue=t.updateQueue,e.flags&=-2053,t.lanes&=~n}function $m(t){if(Xu){for(t=t.memoizedState;t!==null;){var e=t.queue;e!==null&&(e.pending=null),t=t.next}Xu=!1}Ds=0,kt=pt=ze=null,to=!1,zl=Yu=0,no=null}function Ln(){var t={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return kt===null?ze.memoizedState=kt=t:kt=kt.next=t,kt}function Ht(){if(pt===null){var t=ze.alternate;t=t!==null?t.memoizedState:null}else t=pt.next;var e=kt===null?ze.memoizedState:kt.next;if(e!==null)kt=e,pt=t;else{if(t===null)throw ze.alternate===null?Error(J(467)):Error(J(310));pt=t,t={memoizedState:pt.memoizedState,baseState:pt.baseState,baseQueue:pt.baseQueue,queue:pt.queue,next:null},kt===null?ze.memoizedState=kt=t:kt=kt.next=t}return kt}function vf(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function ec(t){var e=zl;return zl+=1,no===null&&(no=[]),t=T_(no,t,e),e=ze,(kt===null?e.memoizedState:kt.next)===null&&(e=e.alternate,Ue.H=e===null||e.memoizedState===null?oS:lS),t}function xf(t){if(t!==null&&typeof t=="object"){if(typeof t.then=="function")return ec(t);if(t.$$typeof===_E)return;if(t.$$typeof===ns)return vn(t)}throw Error(J(438,String(t)))}function eg(t){var e=null,n=ze.updateQueue;if(n!==null&&(e=n.memoCache),e==null){var i=ze.alternate;i!==null&&(i=i.updateQueue,i!==null&&(i=i.memoCache,i!=null&&(e={data:i.data.map(function(s){return s.slice()}),index:0})))}if(e==null&&(e={data:[],index:0}),n===null&&(n=vf(),ze.updateQueue=n),n.memoCache=e,n=e.data[e.index],n===void 0)for(n=e.data[e.index]=Array(t),i=0;i<t;i++)n[i]=yE;return e.index++,n}function Us(t,e){return typeof e=="function"?e(t):e}function _u(t){var e=Ht();return tg(e,pt,t)}function tg(t,e,n){var i=t.queue;if(i===null)throw Error(J(311));i.lastRenderedReducer=n;var s=t.baseQueue,r=i.pending;if(r!==null){if(s!==null){var a=s.next;s.next=r.next,r.next=a}e.baseQueue=s=r,i.pending=null}if(r=t.baseState,s===null)t.memoizedState=r;else{e=s.next;var o=a=null,l=null,c=e,h=!1;do{var p=c.lane&-536870913;if(p!==c.lane?(Je&p)===p:(Ds&p)===p){var u=c.revertLane;if(u===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null}),p===aa&&(h=!0);else if((Ds&u)===u){c=c.next,u===aa&&(h=!0);continue}else p={lane:0,revertLane:c.revertLane,gesture:null,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=p,a=r):l=l.next=p,ze.lanes|=u,Mr|=u;p=c.action,la&&n(r,p),r=c.hasEagerState?c.eagerState:n(r,p)}else u={lane:p,revertLane:c.revertLane,gesture:c.gesture,action:c.action,hasEagerState:c.hasEagerState,eagerState:c.eagerState,next:null},l===null?(o=l=u,a=r):l=l.next=u,ze.lanes|=p,Mr|=p;c=c.next}while(c!==null&&c!==e);if(l===null?a=r:l.next=o,!si(r,t.memoizedState)&&(Xt=!0,h&&(n=$a,n!==null)))throw n;t.memoizedState=r,t.baseState=a,t.baseQueue=l,i.lastRenderedState=r}return s===null&&(i.lanes=0),[t.memoizedState,i.dispatch]}function np(t){var e=Ht(),n=e.queue;if(n===null)throw Error(J(311));n.lastRenderedReducer=t;var i=n.dispatch,s=n.pending,r=e.memoizedState;if(s!==null){n.pending=null;var a=s=s.next;do r=t(r,a.action),a=a.next;while(a!==s);si(r,e.memoizedState)||(Xt=!0),e.memoizedState=r,e.baseQueue===null&&(e.baseState=r),n.lastRenderedState=r}return[r,i]}function B_(t,e,n){var i=ze,s=Ht(),r=ke;if(r){if(n===void 0)throw Error(J(407));n=n()}else n=e();var a=!si((pt||s).memoizedState,n);if(a&&(s.memoizedState=n,Xt=!0),s=s.queue,ng(P_.bind(null,i,s,t),[t]),t=s.getSnapshot!==e||a||kt!==null&&(kt.memoizedState.tag&1)!==0,co(t?9:8,{destroy:void 0},L_.bind(null,i,s,n,e),null),t){if(i.flags|=2048,vt===null)throw Error(J(349));r||(Ds&127)!==0||N_(i,e,n)}return n}function N_(t,e,n){t.flags|=16384,t={getSnapshot:e,value:n},e=ze.updateQueue,e===null?(e=vf(),ze.updateQueue=e,e.stores=[t]):(n=e.stores,n===null?e.stores=[t]:n.push(t))}function L_(t,e,n,i){e.value=n,e.getSnapshot=i,O_(e)&&F_(t)}function P_(t,e,n){return n(function(){O_(e)&&F_(t)})}function O_(t){var e=t.getSnapshot;t=t.value;try{var n=e();return!si(t,n)}catch{return!0}}function F_(t){var e=pa(t,2);e!==null&&Wn(e,t,2)}function Wp(t){var e=Ln();if(typeof t=="function"){var n=t;if(t=n(),la){ir(!0);try{n()}finally{ir(!1)}}}return e.memoizedState=e.baseState=t,e.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Us,lastRenderedState:t},e}function z_(t,e,n,i){return t.baseState=n,tg(t,pt,typeof i=="function"?i:Us)}function HT(t,e,n,i,s){if(_f(t))throw Error(J(485));if(t=e.action,t!==null){var r={payload:s,action:t,next:null,isTransition:!0,status:"pending",value:null,reason:null,listeners:[],then:function(a){r.listeners.push(a)}};Ue.T!==null?n(!0):r.isTransition=!1,i(r),n=e.pending,n===null?(r.next=e.pending=r,H_(e,r)):(r.next=n.next,e.pending=n.next=r)}}function H_(t,e){var n=e.action,i=e.payload,s=t.state;if(e.isTransition){var r=Ue.T,a={};a.types=r!==null?r.types:null,Ue.T=a;try{var o=n(s,i),l=Ue.S;l!==null&&l(a,o),wx(t,e,o)}catch(c){Xp(t,e,c)}finally{r!==null&&a.types!==null&&(r.types=a.types),Ue.T=r}}else try{r=n(s,i),wx(t,e,r)}catch(c){Xp(t,e,c)}}function wx(t,e,n){n!==null&&typeof n=="object"&&typeof n.then=="function"?n.then(function(i){Cx(t,e,i)},function(i){return Xp(t,e,i)}):Cx(t,e,n)}function Cx(t,e,n){e.status="fulfilled",e.value=n,G_(e),t.state=n,e=t.pending,e!==null&&(n=e.next,n===e?t.pending=null:(n=n.next,e.next=n,H_(t,n)))}function Xp(t,e,n){var i=t.pending;if(t.pending=null,i!==null){i=i.next;do e.status="rejected",e.reason=n,G_(e),e=e.next;while(e!==i)}t.action=null}function G_(t){t=t.listeners;for(var e=0;e<t.length;e++)(0,t[e])()}function V_(t,e){return e}function Rx(t,e){if(ke){var n=vt.formState;if(n!==null){e:{var i=ze;if(ke){if(At){t:{for(var s=At,r=Ei;s.nodeType!==8;){if(!r){s=null;break t}if(s=Ti(s.nextSibling),s===null){s=null;break t}}r=s.data,s=r==="F!"||r==="F"?s:null}if(s){At=Ti(s.nextSibling),i=s.data==="F!";break e}}_r(i)}i=!1}i&&(e=n[0])}}return n=Ln(),n.memoizedState=n.baseState=e,i={pending:null,lanes:0,dispatch:null,lastRenderedReducer:V_,lastRenderedState:e},n.queue=i,n=sS.bind(null,ze,i),i.dispatch=n,i=Wp(!1),r=ag.bind(null,ze,!1,i.queue),i=Ln(),s={state:e,dispatch:null,action:t,pending:null},i.queue=s,n=HT.bind(null,ze,s,r,n),s.dispatch=n,i.memoizedState=t,[e,n,!1]}function Dx(t){var e=Ht();return k_(e,pt,t)}function k_(t,e,n){if(e=tg(t,e,V_)[0],t=_u(Us)[0],typeof e=="object"&&e!==null&&typeof e.then=="function")try{var i=ec(e)}catch(a){throw a===Ao?gf:a}else i=e;e=Ht();var s=e.queue,r=s.dispatch;return n!==e.memoizedState&&(ze.flags|=2048,co(9,{destroy:void 0},GT.bind(null,s,n),null)),[i,r,t]}function GT(t,e){t.action=e}function Ux(t){var e=Ht(),n=pt;if(n!==null)return k_(e,n,t);Ht(),e=e.memoizedState,n=Ht();var i=n.queue.dispatch;return n.memoizedState=t,[e,i,!1]}function co(t,e,n,i){return t={tag:t,create:n,deps:i,inst:e,next:null},e=ze.updateQueue,e===null&&(e=vf(),ze.updateQueue=e),n=e.lastEffect,n===null?e.lastEffect=t.next=t:(i=n.next,n.next=t,t.next=i,e.lastEffect=t),t}function W_(){return Ht().memoizedState}function Su(t,e,n,i){var s=Ln();ze.flags|=t,s.memoizedState=co(1|e,{destroy:void 0},n,i===void 0?null:i)}function yf(t,e,n,i){var s=Ht();i=i===void 0?null:i;var r=s.memoizedState.inst;pt!==null&&i!==null&&Zm(i,pt.memoizedState.deps)?s.memoizedState=co(e,r,n,i):(ze.flags|=t,s.memoizedState=co(1|e,r,n,i))}function Ix(t,e){Su(8390656,8,t,e)}function ng(t,e){yf(2048,8,t,e)}function VT(t){ze.flags|=4;var e=ze.updateQueue;if(e===null)e=vf(),ze.updateQueue=e,e.events=[t];else{var n=e.events;n===null?e.events=[t]:n.push(t)}}function X_(t){var e=Ht().memoizedState;return VT({ref:e,nextImpl:t}),function(){if((at&2)!==0)throw Error(J(440));return e.impl.apply(void 0,arguments)}}function Y_(t,e){return yf(4,2,t,e)}function q_(t,e){return yf(4,4,t,e)}function Q_(t,e){if(typeof e=="function"){t=t();var n=e(t);return function(){typeof n=="function"?n():e(null)}}if(e!=null)return t=t(),e.current=t,function(){e.current=null}}function Z_(t,e,n){n=n!=null?n.concat([t]):null,yf(4,4,Q_.bind(null,e,t),n)}function ig(){}function K_(t,e){var n=Ht();e=e===void 0?null:e;var i=n.memoizedState;return e!==null&&Zm(e,i[1])?i[0]:(n.memoizedState=[t,e],t)}function J_(t,e){var n=Ht();e=e===void 0?null:e;var i=n.memoizedState;if(e!==null&&Zm(e,i[1]))return i[0];if(i=t(),la){ir(!0);try{t()}finally{ir(!1)}}return n.memoizedState=[i,e],i}function sg(t,e,n){return n===void 0||(Ds&1073741824)!==0&&(Je&261930)===0?t.memoizedState=e:(t.memoizedState=n,t=kS(),ze.lanes|=t,Mr|=t,n)}function j_(t,e,n,i){return si(n,e)?n:Sr.current!==null?(t=sg(t,n,i),si(t,e)||(Xt=!0),t):(Ds&106)===0||(Ds&1073741824)!==0&&(Je&261930)===0?(Xt=!0,t.memoizedState=n):(t=kS(),ze.lanes|=t,Mr|=t,e)}function $_(t,e,n,i,s){var r=ot.p;ot.p=r!==0&&8>r?r:8;var a=Ue.T,o={};o.types=a!==null?a.types:null,Ue.T=o,ag(t,!1,e,n);try{var l=s(),c=Ue.S;if(c!==null&&c(o,l),l!==null&&typeof l=="object"&&typeof l.then=="function"){var h=OT(l,i);Tl(t,e,h,ii(t))}else Tl(t,e,i,ii(t))}catch(p){Tl(t,e,{then:function(){},status:"rejected",reason:p},ii())}finally{ot.p=r,a!==null&&o.types!==null&&(a.types=o.types),Ue.T=a}}function kT(){}function Yp(t,e,n,i){if(t.tag!==5)throw Error(J(476));var s=eS(t).queue;$_(t,s,e,jr,n===null?kT:function(){return tS(t),n(i)})}function eS(t){var e=t.memoizedState;if(e!==null)return e;e={memoizedState:jr,baseState:jr,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:Us,lastRenderedState:jr},next:null};var n={};return e.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:Us,lastRenderedState:n},next:null},t.memoizedState=e,t=t.alternate,t!==null&&(t.memoizedState=e),e}function tS(t){var e=eS(t);e.next===null&&(e=t.alternate.memoizedState),Tl(t,e.next.queue,{},ii())}function rg(){return vn(vo)}function nS(){return Ht().memoizedState}function iS(){return Ht().memoizedState}function WT(t){for(var e=t.return;e!==null;){switch(e.tag){case 24:case 3:var n=ii();t=fr(n);var i=hr(e,t,n);i!==null&&(Wn(i,e,n),Al(i,e,n)),e={cache:km()},t.payload=e;return}e=e.return}}function XT(t,e,n){var i=ii();n={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},_f(t)?rS(e,n):(n=Hm(t,e,n,i),n!==null&&(Wn(n,t,i),aS(n,e,i)))}function sS(t,e,n){var i=ii();Tl(t,e,n,i)}function Tl(t,e,n,i){var s={lane:i,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(_f(t))rS(e,s);else{var r=t.alternate;if(t.lanes===0&&(r===null||r.lanes===0)&&(r=e.lastRenderedReducer,r!==null))try{var a=e.lastRenderedState,o=r(a,n);if(s.hasEagerState=!0,s.eagerState=o,si(o,a))return pf(t,e,s,0),vt===null&&df(),!1}catch{}if(n=Hm(t,e,s,i),n!==null)return Wn(n,t,i),aS(n,e,i),!0}return!1}function ag(t,e,n,i){if(i={lane:2,revertLane:vg(),gesture:null,action:i,hasEagerState:!1,eagerState:null,next:null},_f(t)){if(e)throw Error(J(479))}else e=Hm(t,n,i,2),e!==null&&Wn(e,t,2)}function _f(t){var e=t.alternate;return t===ze||e!==null&&e===ze}function rS(t,e){to=Xu=!0;var n=t.pending;n===null?e.next=e:(e.next=n.next,n.next=e),t.pending=e}function aS(t,e,n){if((n&4194048)!==0){var i=e.lanes;i&=t.pendingLanes,n|=i,e.lanes=n,Wy(t,n)}}var qu={readContext:vn,use:xf,useCallback:Pt,useContext:Pt,useEffect:Pt,useImperativeHandle:Pt,useLayoutEffect:Pt,useInsertionEffect:Pt,useMemo:Pt,useReducer:Pt,useRef:Pt,useState:Pt,useDebugValue:Pt,useDeferredValue:Pt,useTransition:Pt,useSyncExternalStore:Pt,useId:Pt,useHostTransitionStatus:Pt,useFormState:Pt,useActionState:Pt,useOptimistic:Pt,useMemoCache:Pt,useCacheRefresh:Pt,useEffectEvent:Pt},oS={readContext:vn,use:xf,useCallback:function(t,e){return Ln().memoizedState=[t,e===void 0?null:e],t},useContext:vn,useEffect:Ix,useImperativeHandle:function(t,e,n){n=n!=null?n.concat([t]):null,Su(4194308,4,Q_.bind(null,e,t),n)},useLayoutEffect:function(t,e){return Su(4194308,4,t,e)},useInsertionEffect:function(t,e){Su(4,2,t,e)},useMemo:function(t,e){var n=Ln();e=e===void 0?null:e;var i=t();if(la){ir(!0);try{t()}finally{ir(!1)}}return n.memoizedState=[i,e],i},useReducer:function(t,e,n){var i=Ln();if(n!==void 0){var s=n(e);if(la){ir(!0);try{n(e)}finally{ir(!1)}}}else s=e;return i.memoizedState=i.baseState=s,t={pending:null,lanes:0,dispatch:null,lastRenderedReducer:t,lastRenderedState:s},i.queue=t,t=t.dispatch=XT.bind(null,ze,t),[i.memoizedState,t]},useRef:function(t){var e=Ln();return t={current:t},e.memoizedState=t},useState:function(t){t=Wp(t);var e=t.queue,n=sS.bind(null,ze,e);return e.dispatch=n,[t.memoizedState,n]},useDebugValue:ig,useDeferredValue:function(t,e){var n=Ln();return sg(n,t,e)},useTransition:function(){var t=Wp(!1);return t=$_.bind(null,ze,t.queue,!0,!1),Ln().memoizedState=t,[!1,t]},useSyncExternalStore:function(t,e,n){var i=ze,s=Ln();if(ke){if(n===void 0)throw Error(J(407));n=n()}else{if(n=e(),vt===null)throw Error(J(349));(Je&127)!==0||N_(i,e,n)}s.memoizedState=n;var r={value:n,getSnapshot:e};return s.queue=r,Ix(P_.bind(null,i,r,t),[t]),i.flags|=2048,co(9,{destroy:void 0},L_.bind(null,i,r,n,e),null),n},useId:function(){var t=Ln(),e=vt.identifierPrefix;if(ke){var n=rs,i=ss;n=(i&~(1<<32-ni(i)-1)).toString(32)+n,e="_"+e+"R_"+n,n=Yu++,0<n&&(e+="H"+n.toString(32)),e+="_"}else n=FT++,e="_"+e+"r_"+n.toString(32)+"_";return t.memoizedState=e},useHostTransitionStatus:rg,useFormState:Rx,useActionState:Rx,useOptimistic:function(t){var e=Ln();e.memoizedState=e.baseState=t;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return e.queue=n,e=ag.bind(null,ze,!0,n),n.dispatch=e,[t,e]},useMemoCache:eg,useCacheRefresh:function(){return Ln().memoizedState=WT.bind(null,ze)},useEffectEvent:function(t){var e=Ln(),n={impl:t};return e.memoizedState=n,function(){if((at&2)!==0)throw Error(J(440));return n.impl.apply(void 0,arguments)}}},lS={readContext:vn,use:xf,useCallback:K_,useContext:vn,useEffect:ng,useImperativeHandle:Z_,useInsertionEffect:Y_,useLayoutEffect:q_,useMemo:J_,useReducer:_u,useRef:W_,useState:function(){return _u(Us)},useDebugValue:ig,useDeferredValue:function(t,e){var n=Ht();return j_(n,pt.memoizedState,t,e)},useTransition:function(){var t=_u(Us)[0],e=Ht().memoizedState;return[typeof t=="boolean"?t:ec(t),e]},useSyncExternalStore:B_,useId:nS,useHostTransitionStatus:rg,useFormState:Dx,useActionState:Dx,useOptimistic:function(t,e){var n=Ht();return z_(n,pt,t,e)},useMemoCache:eg,useCacheRefresh:iS,useEffectEvent:X_},YT={readContext:vn,use:xf,useCallback:K_,useContext:vn,useEffect:ng,useImperativeHandle:Z_,useInsertionEffect:Y_,useLayoutEffect:q_,useMemo:J_,useReducer:np,useRef:W_,useState:function(){return np(Us)},useDebugValue:ig,useDeferredValue:function(t,e){var n=Ht();return pt===null?sg(n,t,e):j_(n,pt.memoizedState,t,e)},useTransition:function(){var t=np(Us)[0],e=Ht().memoizedState;return[typeof t=="boolean"?t:ec(t),e]},useSyncExternalStore:B_,useId:nS,useHostTransitionStatus:rg,useFormState:Ux,useActionState:Ux,useOptimistic:function(t,e){var n=Ht();return pt!==null?z_(n,pt,t,e):(n.baseState=t,[t,n.queue.dispatch])},useMemoCache:eg,useCacheRefresh:iS,useEffectEvent:X_};function ip(t,e,n,i){e=t.memoizedState,n=n(i,e),n=n==null?e:xt({},e,n),t.memoizedState=n,t.lanes===0&&(t.updateQueue.baseState=n)}var qp={enqueueSetState:function(t,e,n){t=t._reactInternals;var i=ii(),s=fr(i);s.payload=e,n!=null&&(s.callback=n),e=hr(t,s,i),e!==null&&(Wn(e,t,i),Al(e,t,i))},enqueueReplaceState:function(t,e,n){t=t._reactInternals;var i=ii(),s=fr(i);s.tag=1,s.payload=e,n!=null&&(s.callback=n),e=hr(t,s,i),e!==null&&(Wn(e,t,i),Al(e,t,i))},enqueueForceUpdate:function(t,e){t=t._reactInternals;var n=ii(),i=fr(n);i.tag=2,e!=null&&(i.callback=e),e=hr(t,i,n),e!==null&&(Wn(e,t,n),Al(e,t,n))}};function Bx(t,e,n,i,s,r,a){return t=t.stateNode,typeof t.shouldComponentUpdate=="function"?t.shouldComponentUpdate(i,r,a):e.prototype&&e.prototype.isPureReactComponent?!Nl(n,i)||!Nl(s,r):!0}function Nx(t,e,n,i){t=e.state,typeof e.componentWillReceiveProps=="function"&&e.componentWillReceiveProps(n,i),typeof e.UNSAFE_componentWillReceiveProps=="function"&&e.UNSAFE_componentWillReceiveProps(n,i),e.state!==t&&qp.enqueueReplaceState(e,e.state,null)}function ca(t,e){var n=e;if("ref"in e){n={};for(var i in e)i!=="ref"&&(n[i]=e[i])}if(t=t.defaultProps){n===e&&(n=xt({},n));for(var s in t)n[s]===void 0&&(n[s]=t[s])}return n}function cS(t){Ou(t)}function uS(t){console.error(t)}function fS(t){Ou(t)}function Qu(t,e){try{var n=t.onUncaughtError;n(e.value,{componentStack:e.stack})}catch(i){setTimeout(function(){throw i})}}function Lx(t,e,n){try{var i=t.onCaughtError;i(n.value,{componentStack:n.stack,errorBoundary:e.tag===1?e.stateNode:null})}catch(s){setTimeout(function(){throw s})}}function Qp(t,e,n){return n=fr(n),n.tag=3,n.payload={element:null},n.callback=function(){Qu(t,e)},n}function hS(t){return t=fr(t),t.tag=3,t}function dS(t,e,n,i){var s=n.type.getDerivedStateFromError;if(typeof s=="function"){var r=i.value;t.payload=function(){return s(r)},t.callback=function(){Lx(e,n,i)}}var a=n.stateNode;a!==null&&typeof a.componentDidCatch=="function"&&(t.callback=function(){Lx(e,n,i),typeof s!="function"&&(mr===null?mr=new Set([this]):mr.add(this));var o=i.stack;this.componentDidCatch(i.value,{componentStack:o!==null?o:""})})}function qT(t,e,n,i,s){if(n.flags|=32768,i!==null&&typeof i=="object"&&typeof i.then=="function"){if(e=n.alternate,e!==null&&sa(e,n,s,!0),n=Sn.current,n!==null){switch(n.tag){case 31:case 13:case 19:return bn===null?nf():n.alternate===null&&Ot===0&&(Ot=3),n.flags&=-257,n.flags|=65536,n.lanes=s,i===Vu?n.flags|=16384:(e=n.updateQueue,e===null?n.updateQueue=new Set([i]):e.add(i),up(t,i,s)),!1;case 22:return n.flags|=65536,i===Vu?n.flags|=16384:(e=n.updateQueue,e===null?(e={transitions:null,markerInstances:null,retryQueue:new Set([i])},n.updateQueue=e):(n=e.retryQueue,n===null?e.retryQueue=new Set([i]):n.add(i)),up(t,i,s)),!1}throw Error(J(435,n.tag))}return up(t,i,s),nf(),!1}if(ke)return e=Sn.current,e!==null?((e.flags&65536)===0&&(e.flags|=256),e.flags|=65536,e.lanes=s,i!==Pp&&(t=Error(J(422),{cause:i}),Pl(Mi(t,n)))):(i!==Pp&&(e=Error(J(423),{cause:i}),Pl(Mi(e,n))),t=t.current.alternate,t.flags|=65536,s&=-s,t.lanes|=s,i=Mi(i,n),s=Qp(t.stateNode,i,s),tp(t,s),Ot!==4&&(Ot=2)),!1;var r=Error(J(520),{cause:i});if(r=Mi(r,n),Rl===null?Rl=[r]:Rl.push(r),Ot!==4&&(Ot=2),e===null)return!0;i=Mi(i,n),n=e;do{switch(n.tag){case 3:return n.flags|=65536,t=s&-s,n.lanes|=t,t=Qp(n.stateNode,i,t),tp(n,t),!1;case 1:if(e=n.type,r=n.stateNode,(n.flags&128)===0&&(typeof e.getDerivedStateFromError=="function"||r!==null&&typeof r.componentDidCatch=="function"&&(mr===null||!mr.has(r))))return n.flags|=65536,s&=-s,n.lanes|=s,s=hS(s),dS(s,t,n,i),tp(n,s),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var og=Error(J(461)),Xt=!1;function Jt(t,e,n,i){e.child=t===null?w_(e,null,n,i):oa(e,t.child,n,i)}function Px(t,e,n,i,s){n=n.render;var r=e.ref;if("ref"in i){var a={};for(var o in i)o!=="ref"&&(a[o]=i[o])}else a=i;return ra(e),i=Km(t,e,n,a,r,s),o=Jm(),t!==null&&!Xt?(jm(t,e,s),Is(t,e,s)):(ke&&o&&mf(e),e.flags|=1,Jt(t,e,i,s),e.child)}function Ox(t,e,n,i,s){if(t===null){var r=n.type;return typeof r=="function"&&!Gm(r)&&r.defaultProps===void 0&&n.compare===null?(e.tag=15,e.type=r,pS(t,e,r,i,s)):(t=vu(n.type,null,i,e,e.mode,s),t.ref=e.ref,t.return=e,e.child=t)}if(r=t.child,!cg(t,s)){var a=r.memoizedProps;if(n=n.compare,n=n!==null?n:Nl,n(a,i)&&t.ref===e.ref)return Is(t,e,s)}return e.flags|=1,t=bs(r,i),t.ref=e.ref,t.return=e,e.child=t}function pS(t,e,n,i,s){if(t!==null){var r=t.memoizedProps;if(Nl(r,i)&&t.ref===e.ref)if(Xt=!1,e.pendingProps=i=r,cg(t,s))(t.flags&131072)!==0&&(Xt=!0);else return e.lanes=t.lanes,Is(t,e,s)}return Zp(t,e,n,i,s)}function mS(t,e,n,i){var s=i.children,r=t!==null?t.memoizedState:null;if(t===null&&e.stateNode===null&&(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),i.mode==="hidden"){if((e.flags&128)!==0){if(r=r!==null?r.baseLanes|n:n,t!==null){for(i=e.child=t.child,s=0;i!==null;)s=s|i.lanes|i.childLanes,i=i.sibling;i=s&~r}else i=0,e.child=null;return Fx(t,e,r,n,i)}if((n&536870912)!==0)e.memoizedState={baseLanes:0,cachePool:null},t!==null&&yu(e,r!==null?r.cachePool:null),r!==null?bx(e,r):Vp(),D_(e);else return i=e.lanes=536870912,Fx(t,e,r!==null?r.baseLanes|n:n,n,i)}else r!==null?(yu(e,r.cachePool),bx(e,r),pr(),e.memoizedState=null):(t!==null&&yu(e,null),Vp(),pr());return Jt(t,e,s,n),e.child}function bl(t,e){return t!==null&&t.tag===22||e.stateNode!==null||(e.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),e.sibling}function Fx(t,e,n,i,s){var r=Wm();return r=r===null?null:{parent:Wt._currentValue,pool:r},e.memoizedState={baseLanes:n,cachePool:r},t!==null&&yu(e,null),Vp(),D_(e),t!==null&&sa(t,e,i,!0),e.childLanes=s,null}function Au(t,e){return e=Sf({mode:e.mode,children:e.children},t.mode),e.ref=t.ref,t.child=e,e.return=t,e}function zx(t,e,n){return oa(e,t.child,null,n),t=Au(e,e.pendingProps),t.flags|=2,Jn(e),e.memoizedState=null,t}function QT(t,e,n){var i=e.pendingProps,s=(e.flags&128)!==0;if(e.flags&=-129,t===null){if(ke){if(i.mode==="hidden")return t=Au(e,i),e.lanes=536870912,t.memoizedState={baseLanes:0,cachePool:null},bl(null,t);if(kp(e),(t=At)?(t=mA(t,Ei),t=t!==null&&t.data==="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:yr!==null?{id:ss,overflow:rs}:null,retryLane:536870912,hydrationErrors:null},n=__(t),n.return=e,e.child=n,cn=e,At=null)):t=null,t===null)throw _r(e);return e.lanes=536870912,null}return Au(e,i)}var r=t.memoizedState;if(r!==null){var a=r.dehydrated;if(kp(e),s)if(e.flags&256)e.flags&=-257,e=zx(t,e,n);else if(e.memoizedState!==null)e.child=t.child,e.flags|=128,e=null;else throw Error(J(558));else if(Xt||sa(t,e,n,!1),s=(n&t.childLanes)!==0,Xt||s){if(Sr.current===null){if(i=vt,i!==null&&(a=Xy(i,n),a!==0&&a!==r.retryLane))throw r.retryLane=a,pa(t,a),Wn(i,t,a),og;nf()}e=zx(t,e,n)}else t=r.treeContext,At=Ti(a.nextSibling),cn=e,ke=!0,ur=null,Ei=!1,t!==null&&A_(e,t),e=Au(e,i),e.flags|=134221824;return e}return t=bs(t.child,{mode:i.mode,children:i.children}),t.ref=e.ref,e.child=t,t.return=e,t}function La(t,e){var n=e.ref;if(n===null)t!==null&&t.ref!==null&&(e.flags|=4194816);else{if(typeof n!="function"&&typeof n!="object")throw Error(J(284));(t===null||t.ref!==n)&&(e.flags|=4194816)}}function Zp(t,e,n,i,s){return ra(e),n=Km(t,e,n,i,void 0,s),i=Jm(),t!==null&&!Xt?(jm(t,e,s),Is(t,e,s)):(ke&&i&&mf(e),e.flags|=1,Jt(t,e,n,s),e.child)}function Hx(t,e,n,i,s,r){return ra(e),e.updateQueue=null,n=I_(e,i,n,s),U_(t),i=Jm(),t!==null&&!Xt?(jm(t,e,r),Is(t,e,r)):(ke&&i&&mf(e),e.flags|=1,Jt(t,e,n,r),e.child)}function Gx(t,e,n,i,s){if(ra(e),e.stateNode===null){var r=Ya,a=n.contextType;typeof a=="object"&&a!==null&&(r=vn(a)),r=new n(i,r),e.memoizedState=r.state!==null&&r.state!==void 0?r.state:null,r.updater=qp,e.stateNode=r,r._reactInternals=e,r=e.stateNode,r.props=i,r.state=e.memoizedState,r.refs={},Ym(e),a=n.contextType,r.context=typeof a=="object"&&a!==null?vn(a):Ya,r.state=e.memoizedState,a=n.getDerivedStateFromProps,typeof a=="function"&&(ip(e,n,a,i),r.state=e.memoizedState),typeof n.getDerivedStateFromProps=="function"||typeof r.getSnapshotBeforeUpdate=="function"||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(a=r.state,typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount(),a!==r.state&&qp.enqueueReplaceState(r,r.state,null),El(e,i,r,s),Ml(),r.state=e.memoizedState),typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!0}else if(t===null){r=e.stateNode;var o=e.memoizedProps,l=ca(n,o);r.props=l;var c=r.context,h=n.contextType;a=Ya,typeof h=="object"&&h!==null&&(a=vn(h));var p=n.getDerivedStateFromProps;h=typeof p=="function"||typeof r.getSnapshotBeforeUpdate=="function",o=e.pendingProps!==o,h||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(o||c!==a)&&Nx(e,r,i,a),tr=!1;var u=e.memoizedState;r.state=u,El(e,i,r,s),Ml(),c=e.memoizedState,o||u!==c||tr?(typeof p=="function"&&(ip(e,n,p,i),c=e.memoizedState),(l=tr||Bx(e,n,l,i,u,c,a))?(h||typeof r.UNSAFE_componentWillMount!="function"&&typeof r.componentWillMount!="function"||(typeof r.componentWillMount=="function"&&r.componentWillMount(),typeof r.UNSAFE_componentWillMount=="function"&&r.UNSAFE_componentWillMount()),typeof r.componentDidMount=="function"&&(e.flags|=4194308)):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),e.memoizedProps=i,e.memoizedState=c),r.props=i,r.state=c,r.context=a,i=l):(typeof r.componentDidMount=="function"&&(e.flags|=4194308),i=!1)}else{r=e.stateNode,Hp(t,e),a=e.memoizedProps,h=ca(n,a),r.props=h,p=e.pendingProps,u=r.context,c=n.contextType,l=Ya,typeof c=="object"&&c!==null&&(l=vn(c)),o=n.getDerivedStateFromProps,(c=typeof o=="function"||typeof r.getSnapshotBeforeUpdate=="function")||typeof r.UNSAFE_componentWillReceiveProps!="function"&&typeof r.componentWillReceiveProps!="function"||(a!==p||u!==l)&&Nx(e,r,i,l),tr=!1,u=e.memoizedState,r.state=u,El(e,i,r,s),Ml();var d=e.memoizedState;a!==p||u!==d||tr||t!==null&&t.dependencies!==null&&Gu(t.dependencies)?(typeof o=="function"&&(ip(e,n,o,i),d=e.memoizedState),(h=tr||Bx(e,n,h,i,u,d,l)||t!==null&&t.dependencies!==null&&Gu(t.dependencies))?(c||typeof r.UNSAFE_componentWillUpdate!="function"&&typeof r.componentWillUpdate!="function"||(typeof r.componentWillUpdate=="function"&&r.componentWillUpdate(i,d,l),typeof r.UNSAFE_componentWillUpdate=="function"&&r.UNSAFE_componentWillUpdate(i,d,l)),typeof r.componentDidUpdate=="function"&&(e.flags|=4),typeof r.getSnapshotBeforeUpdate=="function"&&(e.flags|=1024)):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),e.memoizedProps=i,e.memoizedState=d),r.props=i,r.state=d,r.context=l,i=h):(typeof r.componentDidUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=4),typeof r.getSnapshotBeforeUpdate!="function"||a===t.memoizedProps&&u===t.memoizedState||(e.flags|=1024),i=!1)}return r=i,La(t,e),i=(e.flags&128)!==0,r||i?(r=e.stateNode,n=i&&typeof n.getDerivedStateFromError!="function"?null:r.render(),e.flags|=1,t!==null&&i?(e.child=oa(e,t.child,null,s),e.child=oa(e,null,n,s)):Jt(t,e,n,s),e.memoizedState=r.state,t=e.child):t=Is(t,e,s),t}function Vx(t,e,n,i){return ia(),e.flags|=256,Jt(t,e,n,i),e.child}var Kp={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function Jp(t){return{baseLanes:t,cachePool:E_()}}function jp(t,e,n){return t=t!==null?t.childLanes&~n:0,e&&(t|=$n),t}function gS(t,e,n){var i=e.pendingProps,s=!1,r=(e.flags&128)!==0,a;if((a=r)||(a=t!==null&&t.memoizedState===null?!1:(yn.current&2)!==0),a&&(s=!0,e.flags&=-129),a=(e.flags&32)!==0,e.flags&=-33,t===null){if(ke){if(s?dr(e):pr(),(t=At)?(t=mA(t,Ei),t=t!==null&&t.data!=="&"?t:null,t!==null&&(e.memoizedState={dehydrated:t,treeContext:yr!==null?{id:ss,overflow:rs}:null,retryLane:536870912,hydrationErrors:null},n=__(t),n.return=e,e.child=n,cn=e,At=null)):t=null,t===null)throw _r(e);return Sg(t)?e.lanes=32:e.lanes=536870912,null}return r=i.children,i=i.fallback,s?(pr(),s=e.mode,r=Sf({mode:"hidden",children:r},s),i=$r(i,s,n,null),r.return=e,i.return=e,r.sibling=i,e.child=r,i=e.child,i.memoizedState=Jp(n),i.childLanes=jp(t,a,n),e.memoizedState=Kp,bl(null,i)):(dr(e),lg(e,r))}var o=t.memoizedState;if(o!==null){var l=o.dehydrated;if(l!==null)return ZT(t,e,r,a,i,l,o,n)}return s?(pr(),s=i.fallback,r=e.mode,o=t.child,l=o.sibling,i=bs(o,{mode:"hidden",children:i.children}),i.subtreeFlags=o.subtreeFlags&1206910976,l!==null?s=bs(l,s):(s=$r(s,r,n,null),s.flags|=2),s.return=e,i.return=e,i.sibling=s,e.child=i,bl(null,i),i=e.child,s=t.child.memoizedState,s===null?s=Jp(n):(r=s.cachePool,r!==null?(o=Wt._currentValue,r=r.parent!==o?{parent:o,pool:o}:r):r=E_(),s={baseLanes:s.baseLanes|n,cachePool:r}),i.memoizedState=s,i.childLanes=jp(t,a,n),e.memoizedState=Kp,bl(t.child,i)):(dr(e),n=t.child,t=n.sibling,n=bs(n,{mode:"visible",children:i.children}),n.return=e,n.sibling=null,t!==null&&(a=e.deletions,a===null?(e.deletions=[t],e.flags|=16):a.push(t)),e.child=n,e.memoizedState=null,n)}function lg(t,e){return e=Sf({mode:"visible",children:e},t.mode),e.return=t,t.child=e}function Sf(t,e){return t=kn(22,t,null,e),t.lanes=0,t}function iu(t,e,n){return oa(e,t.child,null,n),t=lg(e,e.pendingProps.children),t.flags|=2,e.memoizedState=null,t}function ZT(t,e,n,i,s,r,a,o){if(n)return e.flags&256?(dr(e),e.flags&=-257,iu(t,e,o)):e.memoizedState!==null?(pr(),e.child=t.child,e.flags|=128,null):(pr(),r=s.fallback,a=e.mode,s=Sf({mode:"visible",children:s.children},a),r=$r(r,a,o,null),r.flags|=2,s.return=e,r.return=e,s.sibling=r,e.child=s,oa(e,t.child,null,o),s=e.child,s.memoizedState=Jp(o),s.childLanes=jp(t,i,o),e.memoizedState=Kp,bl(null,s));if(dr(e),Sg(r)){if(i=r.nextSibling&&r.nextSibling.dataset,i)var l=i.dgst;return i=l,i!==""&&(s=Error(J(419)),s.stack="",s.digest=i,Pl({value:s,source:null,stack:null})),iu(t,e,o)}if(Xt||sa(t,e,o,!1),i=(o&t.childLanes)!==0,Xt||i){if(Sr.current!==null)return iu(t,e,o);if(i=vt,i!==null&&(s=Xy(i,o),s!==0&&s!==a.retryLane))throw a.retryLane=s,pa(t,s),Wn(i,t,s),og;return Mm(r)||nf(),iu(t,e,o)}return Mm(r)?(e.flags|=192,e.child=t.child,null):(t=a.treeContext,At=Ti(r.nextSibling),cn=e,ke=!0,ur=null,Ei=!1,t!==null&&A_(e,t),e=lg(e,s.children),e.flags|=134221824,e)}function kx(t,e,n){t.lanes|=e;var i=t.alternate;i!==null&&(i.lanes|=e),xu(t.return,e,n)}function Wx(t){for(var e=null;t!==null;){var n=t.alternate;n!==null&&Wu(n)===null&&(e=t),t=t.sibling}return e}function su(t,e,n,i,s,r){var a=t.memoizedState;a===null?t.memoizedState={isBackwards:e,rendering:null,renderingStartTime:0,last:i,tail:n,tailMode:s,treeForkCount:r}:(a.isBackwards=e,a.rendering=null,a.renderingStartTime=0,a.last=i,a.tail=n,a.tailMode=s,a.treeForkCount=r)}function sp(t){var e=t.child;for(t.child=null;e!==null;){var n=e.sibling;e.sibling=t.child,t.child=e,e=n}}function $p(t,e,n){var i=e.pendingProps,s=i.revealOrder,r=i.tail;i=i.children;var a=yn.current;if(e.flags&128)return Fl(e,a),null;var o=(a&2)!==0;if(o?(a=a&1|2,e.flags|=128):a&=1,Fl(e,a),s==="backwards"&&t!==null?(sp(t),Jt(t,e,i,n),sp(t)):Jt(t,e,i,n),i=ke?Ll:0,!o&&t!==null&&(t.flags&128)!==0)e:for(t=e.child;t!==null;){if(t.tag===13)t.memoizedState!==null&&kx(t,n,e);else if(t.tag===19)kx(t,n,e);else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break e;for(;t.sibling===null;){if(t.return===null||t.return===e)break e;t=t.return}t.sibling.return=t.return,t=t.sibling}switch(s){case"backwards":n=Wx(e.child),n===null?(s=e.child,e.child=null):(s=n.sibling,n.sibling=null,sp(e)),su(e,!0,s,null,r,i);break;case"unstable_legacy-backwards":for(n=null,s=e.child,e.child=null;s!==null;){if(t=s.alternate,t!==null&&Wu(t)===null){e.child=s;break}t=s.sibling,s.sibling=n,n=s,s=t}su(e,!0,n,null,r,i);break;case"together":su(e,!1,null,null,void 0,i);break;case"independent":e.memoizedState=null;break;default:n=Wx(e.child),n===null?(s=e.child,e.child=null):(s=n.sibling,n.sibling=null),su(e,!1,s,n,r,i)}return e.child}function Xx(t,e,n){var i=e.pendingProps;return rr(e,e.type,i.value),Jt(t,e,i.children,n),e.child}function Is(t,e,n){if(t!==null&&(e.dependencies=t.dependencies),Mr|=e.lanes,(n&e.childLanes)===0)if(t!==null){if(sa(t,e,n,!1),(n&e.childLanes)===0)return null}else return null;if(t!==null&&e.child!==t.child)throw Error(J(153));if(e.child!==null){for(t=e.child,n=bs(t,t.pendingProps),e.child=n,n.return=e;t.sibling!==null;)t=t.sibling,n=n.sibling=bs(t,t.pendingProps),n.return=e;n.sibling=null}return e.child}function cg(t,e){return(t.lanes&e)!==0?!0:(t=t.dependencies,!!(t!==null&&Gu(t)))}function KT(t,e,n){switch(e.tag){case 3:Bu(e,e.stateNode.containerInfo),rr(e,Wt,t.memoizedState.cache),ia();break;case 27:case 5:bp(e);break;case 4:Bu(e,e.stateNode.containerInfo);break;case 10:rr(e,e.type,e.memoizedProps.value);break;case 31:if(e.memoizedState!==null)return e.flags|=128,kp(e),null;break;case 13:var i=e.memoizedState;if(i!==null){if(i.dehydrated!==null)return dr(e),e.flags|=128,null;i=sa(t,e,n,!1);var s=e.child.childLanes;return i||(n&s)!==0?gS(t,e,n):(dr(e),t=Is(t,e,n),t!==null?t.sibling:null)}dr(e);break;case 19:if(e.flags&128)return $p(t,e,n);if(s=(t.flags&128)!==0,i=(n&e.childLanes)!==0,i||(sa(t,e,n,!1),i=(n&e.childLanes)!==0),s){if(i)return $p(t,e,n);e.flags|=128}if(s=e.memoizedState,s!==null&&(s.rendering=null,s.tail=null,s.lastEffect=null),Fl(e,yn.current),i)break;return null;case 22:return e.lanes=0,mS(t,e,n,e.pendingProps);case 24:rr(e,Wt,t.memoizedState.cache)}return Is(t,e,n)}function vS(t,e,n){if(t!==null)if(t.memoizedProps!==e.pendingProps)Xt=!0;else{if(!cg(t,n)&&(e.flags&128)===0)return Xt=!1,KT(t,e,n);Xt=(t.flags&131072)!==0}else Xt=!1,ke&&(e.flags&1048576)!==0&&S_(e,Ll,e.index);switch(e.lanes=0,e.tag){case 16:e:{var i=e.pendingProps;if(t=Qr(e.elementType),e.type=t,typeof t=="function")Gm(t)?(i=ca(t,i),e.tag=1,e=Gx(null,e,t,i,n)):(e.tag=0,e=Zp(null,e,t,i,n));else{if(t!=null){var s=t.$$typeof;if(s===Cm){e.tag=11,e=Px(null,e,t,i,n);break e}else if(s===Rm){e.tag=14,e=Ox(null,e,t,i,n);break e}else if(s===ns){e.tag=10,e.type=t,e=Xx(null,e,n);break e}}throw e=Ep(t)||t,Error(J(306,e,""))}}return e;case 0:return Zp(t,e,e.type,e.pendingProps,n);case 1:return i=e.type,s=ca(i,e.pendingProps),Gx(t,e,i,s,n);case 3:e:{if(Bu(e,e.stateNode.containerInfo),t===null)throw Error(J(387));i=e.pendingProps;var r=e.memoizedState;s=r.element,Hp(t,e),El(e,i,null,n);var a=e.memoizedState;if(i=a.cache,rr(e,Wt,i),i!==r.cache&&Fp(e,[Wt],n,!0),Ml(),i=a.element,r.isDehydrated)if(r={element:i,isDehydrated:!1,cache:a.cache},e.updateQueue.baseState=r,e.memoizedState=r,e.flags&256){e=Vx(t,e,i,n);break e}else if(i!==s){s=Mi(Error(J(424)),e),Pl(s),e=Vx(t,e,i,n);break e}else for(t=e.stateNode.containerInfo,t.nodeType===9?t=t.body:t=t.nodeName==="HTML"?t.ownerDocument.body:t,At=Ti(t.firstChild),cn=e,ke=!0,ur=null,Ei=!0,n=w_(e,null,i,n),e.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling;else{if(ia(),i===s){e=Is(t,e,n);break e}Jt(t,e,i,n)}e=e.child}return e;case 26:return La(t,e),t===null?(n=xy(e.type,null,e.pendingProps,null))?e.memoizedState=n:ke||(e.stateNode=aA(e.type,e.pendingProps,cr.current,e)):e.memoizedState=xy(e.type,t.memoizedProps,e.pendingProps,t.memoizedState),null;case 27:return bp(e),t===null&&ke&&(i=e.stateNode=gA(e.type,e.pendingProps,cr.current),cn=e,Ei=!0,s=At,Tr(e.type)?(Em=s,At=Ti(i.firstChild)):At=s),Jt(t,e,e.pendingProps.children,n),La(t,e),t===null&&(e.flags|=4194304),e.child;case 5:return t===null&&ke&&((s=i=At)&&(i=Vb(i,e.type,e.pendingProps,Ei),i!==null?(e.stateNode=i,cn=e,At=Ti(i.firstChild),Ei=!1,s=!0):s=!1),s||_r(e)),bp(e),s=e.type,r=e.pendingProps,a=t!==null?t.memoizedProps:null,i=r.children,_m(s,r)?i=null:a!==null&&_m(s,a)&&(e.flags|=32),e.memoizedState!==null&&(s=Km(t,e,zT,null,null,n),vo._currentValue=s),La(t,e),Jt(t,e,i,n),e.child;case 6:return t===null&&ke&&((t=n=At)&&(n=kb(n,e.pendingProps,Ei),n!==null?(e.stateNode=n,cn=e,At=null,t=!0):t=!1),t||_r(e)),null;case 13:return gS(t,e,n);case 4:return Bu(e,e.stateNode.containerInfo),i=e.pendingProps,t===null?e.child=oa(e,null,i,n):Jt(t,e,i,n),e.child;case 11:return Px(t,e,e.type,e.pendingProps,n);case 7:return i=e.pendingProps,La(t,e),Jt(t,e,i,n),e.child;case 8:return Jt(t,e,e.pendingProps.children,n),e.child;case 12:return Jt(t,e,e.pendingProps.children,n),e.child;case 10:return Xx(t,e,n);case 9:return s=e.type._context,i=e.pendingProps.children,ra(e),s=vn(s),i=i(s),e.flags|=1,Jt(t,e,i,n),e.child;case 14:return Ox(t,e,e.type,e.pendingProps,n);case 15:return pS(t,e,e.type,e.pendingProps,n);case 19:return $p(t,e,n);case 31:return QT(t,e,n);case 22:return mS(t,e,n,e.pendingProps);case 24:return ra(e),i=vn(Wt),t===null?(s=Wm(),s===null&&(s=vt,r=km(),s.pooledCache=r,r.refCount++,r!==null&&(s.pooledCacheLanes|=n),s=r),e.memoizedState={parent:i,cache:s},Ym(e),rr(e,Wt,s)):((t.lanes&n)!==0&&(Hp(t,e),El(e,null,null,n),Ml()),s=t.memoizedState,r=e.memoizedState,s.parent!==i?(s={parent:i,cache:i},e.memoizedState=s,e.lanes===0&&(e.memoizedState=e.updateQueue.baseState=s),rr(e,Wt,i)):(i=r.cache,rr(e,Wt,i),i!==s.cache&&Fp(e,[Wt],n,!0))),Jt(t,e,e.pendingProps.children,n),e.child;case 30:return e.stateNode===null&&(e.stateNode={autoName:null,paired:null,clones:null,ref:null}),i=e.pendingProps,i.name!=null&&i.name!=="auto"?e.flags|=t===null?18882560:18874368:ke&&mf(e),t!==null&&t.memoizedProps.name!==i.name?e.flags|=4194816:La(t,e),Jt(t,e,i.children,n),e.child;case 29:throw e.pendingProps}throw Error(J(156,e.tag))}function Ms(t){t.flags|=4}function rp(t,e,n,i,s){var r;if((r=(t.mode&32)!==0)&&(r=n===null?Sy(e,i):Sy(e,i)&&(i.src!==n.src||i.srcSet!==n.srcSet)),r){if(t.flags|=16777216,(s&335544128)===s)if(t.stateNode.complete)t.flags|=8192;else if(YS())t.flags|=8192;else throw ta=Vu,Xm}else t.flags&=-16777217}function Yx(t,e){if(e.type!=="stylesheet"||(e.state.loading&4)!==0)t.flags&=-16777217;else if(t.flags|=16777216,!_A(e))if(YS())t.flags|=8192;else throw ta=Vu,Xm}function ru(t,e){e!==null&&(t.flags|=4),t.flags&16384&&(e=t.tag!==22?Vy():536870912,t.lanes|=e,uo|=e)}function cl(t,e){if(!ke)switch(t.tailMode){case"visible":break;case"collapsed":for(var n=t.tail,i=null;n!==null;)n.alternate!==null&&(i=n),n=n.sibling;i===null?e||t.tail===null?t.tail=null:t.tail.sibling=null:i.sibling=null;break;default:for(e=t.tail,n=null;e!==null;)e.alternate!==null&&(n=e),e=e.sibling;n===null?t.tail=null:n.sibling=null}}function St(t){var e=t.alternate!==null&&t.alternate.child===t.child,n=0,i=0;if(e)for(var s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags&1206910976,i|=s.flags&1206910976,s.return=t,s=s.sibling;else for(s=t.child;s!==null;)n|=s.lanes|s.childLanes,i|=s.subtreeFlags,i|=s.flags,s.return=t,s=s.sibling;return t.subtreeFlags|=i,t.childLanes=n,e}function JT(t,e,n){var i=e.pendingProps;switch(Vm(e),e.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return St(e),null;case 1:return St(e),null;case 3:return n=e.stateNode,i=null,t!==null&&(i=t.memoizedState.cache),e.memoizedState.cache!==i&&(e.flags|=2048),ws(Wt),ao(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(t===null||t.child===null)&&(Ba(e)?Ms(e):t===null||t.memoizedState.isDehydrated&&(e.flags&256)===0||(e.flags|=1024,ep())),St(e),null;case 26:var s=e.type,r=e.memoizedState;return t===null?(Ms(e),r!==null?(St(e),Yx(e,r)):(St(e),rp(e,s,null,i,n))):r?r!==t.memoizedState?(Ms(e),St(e),Yx(e,r)):(St(e),e.flags&=-16777217):(t=t.memoizedProps,t!==i&&Ms(e),St(e),rp(e,s,t,i,n)),null;case 27:if(Nu(e),n=cr.current,s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&Ms(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return St(e),e.subtreeFlags&=-33554433,null}t=as.current,Ba(e)?yx(e,t):(t=gA(s,i,n),e.stateNode=t,Ms(e))}return St(e),e.subtreeFlags&=-33554433,null;case 5:if(Nu(e),s=e.type,t!==null&&e.stateNode!=null)t.memoizedProps!==i&&Ms(e);else{if(!i){if(e.stateNode===null)throw Error(J(166));return St(e),e.subtreeFlags&=-33554433,null}if(r=as.current,Ba(e))yx(e,r);else{var a=Vl(cr.current);switch(r){case 1:r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case 2:r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;default:switch(s){case"svg":r=a.createElementNS("http://www.w3.org/2000/svg",s);break;case"math":r=a.createElementNS("http://www.w3.org/1998/Math/MathML",s);break;case"script":r=a.createElement("div"),r.innerHTML="<script><\/script>",r=r.removeChild(r.firstChild);break;case"select":r=typeof i.is=="string"?a.createElement("select",{is:i.is}):a.createElement("select"),i.multiple?r.multiple=!0:i.size&&(r.size=i.size);break;default:r=typeof i.is=="string"?a.createElement(s,{is:i.is}):a.createElement(s)}}r[gn]=e,r[Yn]=i;e:for(a=e.child;a!==null;){if(a.tag===5||a.tag===6)r.appendChild(a.stateNode);else if(a.tag!==4&&a.tag!==27&&a.child!==null){a.child.return=a,a=a.child;continue}if(a===e)break e;for(;a.sibling===null;){if(a.return===null||a.return===e)break e;a=a.return}a.sibling.return=a.return,a=a.sibling}e.stateNode=r;e:switch(_n(r,s,i),s){case"button":case"input":case"select":case"textarea":i=!!i.autoFocus;break e;case"img":i=!0;break e;default:i=!1}i&&Ms(e)}}return St(e),e.subtreeFlags&=-33554433,rp(e,e.type,t===null?null:t.memoizedProps,e.pendingProps,n),null;case 6:if(t&&e.stateNode!=null)t.memoizedProps!==i&&Ms(e);else{if(typeof i!="string"&&e.stateNode===null)throw Error(J(166));if(t=cr.current,Ba(e)){if(t=e.stateNode,n=e.memoizedProps,i=null,s=cn,s!==null)switch(s.tag){case 27:case 5:i=s.memoizedProps}t[gn]=e,t=!!(t.nodeValue===n||i!==null&&i.suppressHydrationWarning===!0||sA(t.nodeValue,n)),t||_r(e,!0)}else t=Vl(t).createTextNode(i),t[gn]=e,e.stateNode=t}return St(e),null;case 31:if(n=e.memoizedState,t===null||t.memoizedState!==null){if(i=Ba(e),n!==null){if(t===null){if(!i)throw Error(J(318));if(t=e.memoizedState,t=t!==null?t.dehydrated:null,!t)throw Error(J(557));t[gn]=e}else ia(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;St(e),t=!1}else n=ep(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=n),t=!0;if(!t)return e.flags&256?(Jn(e),e):(Jn(e),null);if((e.flags&128)!==0)throw Error(J(558))}return St(e),null;case 13:if(i=e.memoizedState,t===null||t.memoizedState!==null&&t.memoizedState.dehydrated!==null){if(s=Ba(e),i!==null&&i.dehydrated!==null){if(t===null){if(!s)throw Error(J(318));if(s=e.memoizedState,s=s!==null?s.dehydrated:null,!s)throw Error(J(317));s[gn]=e}else ia(),(e.flags&128)===0&&(e.memoizedState=null),e.flags|=4;St(e),s=!1}else s=ep(),t!==null&&t.memoizedState!==null&&(t.memoizedState.hydrationErrors=s),s=!0;if(!s)return e.flags&256?(Jn(e),e):(Jn(e),null)}return Jn(e),(e.flags&128)!==0?(e.lanes=n,e):(n=i!==null,t=t!==null&&t.memoizedState!==null,n&&(i=e.child,s=null,i.alternate!==null&&i.alternate.memoizedState!==null&&i.alternate.memoizedState.cachePool!==null&&(s=i.alternate.memoizedState.cachePool.pool),r=null,i.memoizedState!==null&&i.memoizedState.cachePool!==null&&(r=i.memoizedState.cachePool.pool),r!==s&&(i.flags|=2048)),n!==t&&n&&(e.child.flags|=8192),ru(e,e.updateQueue),St(e),null);case 4:return ao(),t===null&&xg(e.stateNode.containerInfo),e.flags|=67108864,St(e),null;case 10:return ws(e.type),St(e),null;case 19:if(Qm(e),i=e.memoizedState,i===null)return St(e),null;if(s=(e.flags&128)!==0,r=i.rendering,r===null)if(s)cl(i,!1);else{if(Ot!==0||t!==null&&(t.flags&128)!==0)for(t=e.child;t!==null;){if(r=Wu(t),r!==null){for(e.flags|=128,cl(i,!1),t=r.updateQueue,e.updateQueue=t,ru(e,t),e.subtreeFlags=0,t=n,n=e.child;n!==null;)y_(n,t),n=n.sibling;return Fl(e,yn.current&1|2),ke&&Es(e,i.treeForkCount),e.child}t=t.sibling}i.tail!==null&&ei()>ef&&(e.flags|=128,s=!0,cl(i,!1),e.lanes=4194304)}else{if(!s)if(t=Wu(r),t!==null){if(e.flags|=128,s=!0,t=t.updateQueue,e.updateQueue=t,ru(e,t),cl(i,!0),i.tail===null&&i.tailMode!=="collapsed"&&i.tailMode!=="visible"&&!r.alternate&&!ke)return St(e),null}else 2*ei()-i.renderingStartTime>ef&&n!==536870912&&(e.flags|=128,s=!0,cl(i,!1),e.lanes=4194304);i.isBackwards?(r.sibling=e.child,e.child=r):(t=i.last,t!==null?t.sibling=r:e.child=r,i.last=r)}if(i.tail!==null){t=i.tail;e:{for(n=t;n!==null;){if(n.alternate!==null){n=!1;break e}n=n.sibling}n=!0}return i.rendering=t,i.tail=t.sibling,i.renderingStartTime=ei(),t.sibling=null,r=yn.current,r=s?r&1|2:r&1,i.tailMode==="visible"||i.tailMode==="collapsed"||!n||ke?Fl(e,r):(n=r,Mt(Sn,e),Mt(yn,n),bn===null&&(bn=e)),ke&&Es(e,i.treeForkCount),t}return St(e),null;case 22:case 23:return Jn(e),qm(),i=e.memoizedState!==null,t!==null?t.memoizedState!==null!==i&&(e.flags|=8192):i&&(e.flags|=8192),i?(n&536870912)!==0&&(e.flags&128)===0&&(St(e),e.subtreeFlags&6&&(e.flags|=8192)):St(e),n=e.updateQueue,n!==null&&ru(e,n.retryQueue),n=null,t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),i=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(i=e.memoizedState.cachePool.pool),i!==n&&(e.flags|=2048),t!==null&&xn(ea),null;case 24:return n=null,t!==null&&(n=t.memoizedState.cache),e.memoizedState.cache!==n&&(e.flags|=2048),ws(Wt),St(e),null;case 25:return null;case 30:return e.flags|=33554432,St(e),null}throw Error(J(156,e.tag))}function jT(t,e){switch(Vm(e),e.tag){case 1:return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 3:return ws(Wt),ao(),t=e.flags,(t&65536)!==0&&(t&128)===0?(e.flags=t&-65537|128,e):null;case 26:case 27:case 5:return Nu(e),null;case 31:if(e.memoizedState!==null){if(Jn(e),e.alternate===null)throw Error(J(340));ia()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 13:if(Jn(e),t=e.memoizedState,t!==null&&t.dehydrated!==null){if(e.alternate===null)throw Error(J(340));ia()}return t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 19:return Qm(e),t=e.flags,t&65536?(e.flags=t&-65537|128,t=e.memoizedState,t!==null&&(t.rendering=null,t.tail=null),e.flags|=4,e):null;case 4:return ao(),null;case 10:return ws(e.type),null;case 22:case 23:return Jn(e),qm(),t!==null&&xn(ea),t=e.flags,t&65536?(e.flags=t&-65537|128,e):null;case 24:return ws(Wt),null;case 25:return null;default:return null}}function xS(t,e){switch(Vm(e),e.tag){case 3:ws(Wt),ao();break;case 26:case 27:case 5:Nu(e);break;case 4:ao();break;case 31:e.memoizedState!==null&&Jn(e);break;case 13:Jn(e);break;case 19:Qm(e);break;case 10:ws(e.type);break;case 22:case 23:Jn(e),qm(),t!==null&&xn(ea);break;case 24:ws(Wt)}}function tc(t,e){try{var n=e.updateQueue,i=n!==null?n.lastEffect:null;if(i!==null){var s=i.next;n=s;do{if((n.tag&t)===t){i=void 0;var r=n.create,a=n.inst;i=r(),a.destroy=i}n=n.next}while(n!==s)}}catch(o){dt(e,e.return,o)}}function Ar(t,e,n){try{var i=e.updateQueue,s=i!==null?i.lastEffect:null;if(s!==null){var r=s.next;i=r;do{if((i.tag&t)===t){var a=i.inst,o=a.destroy;if(o!==void 0){a.destroy=void 0,s=e;var l=n,c=o;try{c()}catch(h){dt(s,l,h)}}}i=i.next}while(i!==r)}}catch(h){dt(e,e.return,h)}}function yS(t){var e=t.updateQueue;if(e!==null){var n=t.stateNode;try{R_(e,n)}catch(i){dt(t,t.return,i)}}}function _S(t,e,n){n.props=ca(t.type,t.memoizedProps),n.state=t.memoizedState;try{n.componentWillUnmount()}catch(i){dt(t,e,i)}}function es(t,e){try{var n=t.ref;if(n!==null){switch(t.tag){case 26:case 27:case 5:var i=t.stateNode;break;case 30:var s=t.stateNode,r=Rs(t.memoizedProps,s);(s.ref===null||s.ref.name!==r)&&(s.ref=uA(r)),i=s.ref;break;case 7:if(t.stateNode===null){var a=new ri(t);Xn(t.child,!1,Hb,a,void 0,void 0),t.stateNode=a}i=t.stateNode;break;default:i=t.stateNode}typeof n=="function"?t.refCleanup=n(i):n.current=i}}catch(o){dt(t,e,o)}}function mn(t,e){var n=t.ref,i=t.refCleanup;if(n!==null)if(typeof i=="function")try{i()}catch(s){dt(t,e,s)}finally{t.refCleanup=null,t=t.alternate,t!=null&&(t.refCleanup=null)}else if(typeof n=="function")try{n(null)}catch(s){dt(t,e,s)}else n.current=null}function Zu(t,e){if((t.tag===5||t.tag===27||t.tag===6)&&t.alternate===null&&e!==null)for(var n=0;n<e.length;n++)pA(t.stateNode,e[n])}function qx(t){for(var e=t.return;e!==null&&(fg(e)&&pA(t.stateNode,e.stateNode),!ug(e));)e=e.return}function wl(t){for(var e=t.return;e!==null&&(fg(e)&&Gb(t.stateNode,e.stateNode),!ug(e));)e=e.return}function ug(t){return t.tag===5||t.tag===3||t.tag===27}function fg(t){return t&&t.tag===7&&t.stateNode!==null}function em(t){var e=t.type,n=t.memoizedProps,i=t.stateNode;try{e:switch(e){case"button":case"input":case"select":case"textarea":n.autoFocus&&i.focus();break e;case"img":n.src?i.src=n.src:n.srcSet&&(i.srcset=n.srcSet)}}catch(s){dt(t,t.return,s)}}function ap(t,e,n){try{var i=t.stateNode;Ab(i,t.type,n,e),i[Yn]=e}catch(s){dt(t,t.return,s)}}function SS(t){return t.tag===5||t.tag===3||t.tag===26||t.tag===27&&Tr(t.type)||t.tag===4}function op(t){e:for(;;){for(;t.sibling===null;){if(t.return===null||SS(t.return))return null;t=t.return}for(t.sibling.return=t.return,t=t.sibling;t.tag!==5&&t.tag!==6&&t.tag!==18;){if(t.tag===27&&Tr(t.type)||t.flags&2||t.child===null||t.tag===4)continue e;t.child.return=t,t=t.child}if(!(t.flags&2))return t.stateNode}}function tm(t,e,n,i){var s=t.tag;if(s===5||s===6)s=t.stateNode,e?(n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n).insertBefore(s,e):(e=n.nodeType===9?n.body:n.nodeName==="HTML"?n.ownerDocument.body:n,e.appendChild(s),n=n._reactRootContainer,n!=null||e.onclick!==null||(e.onclick=is)),Zu(t,i),nt=!0;else if(s!==4&&(s===27&&(Zu(t,i),i=null,Tr(t.type)&&(n=t.stateNode,e=null)),t=t.child,t!==null))for(tm(t,e,n,i),t=t.sibling;t!==null;)tm(t,e,n,i),t=t.sibling}function Ku(t,e,n,i){var s=t.tag;if(s===5||s===6)s=t.stateNode,e?n.insertBefore(s,e):n.appendChild(s),Zu(t,i),nt=!0;else if(s!==4&&(s===27&&(Zu(t,i),i=null,Tr(t.type)&&(n=t.stateNode)),t=t.child,t!==null))for(Ku(t,e,n,i),t=t.sibling;t!==null;)Ku(t,e,n,i),t=t.sibling}function AS(t){var e=t.stateNode,n=t.memoizedProps;try{for(var i=t.type,s=e.attributes;s.length;)e.removeAttributeNode(s[0]);_n(e,i,n),e[gn]=t,e[Yn]=n}catch(r){dt(t,t.return,r)}}var Ju=!1,jn=null;function Qx(t){(t.tag===30||(t.subtreeFlags&33554432)!==0)&&(Ju=!0)}var ts=null;function Zx(){var t=ts;return ts=null,t}var Vn=0;function Mo(t,e,n,i,s){return Vn=0,MS(t.child,e,n,i,s)}function MS(t,e,n,i,s){for(var r=!1;t!==null;){if(t.tag===5){var a=t.stateNode;if(i!==null){var o=Sm(a);i.push(o),o.view&&(r=!0)}else r||Sm(a).view&&(r=!0);Ju=!0,oA(a,Vn===0?e:e+"_"+Vn,n),Vn++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&s||MS(t.child,e,n,i,s)&&(r=!0));t=t.sibling}return r}function ls(t,e){for(;t!==null;)t.tag===5?lA(t.stateNode,t.memoizedProps):(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&e||ls(t.child,e)),t=t.sibling}function Mu(t){if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if((t.tag!==22||t.memoizedState===null)&&(Mu(t),t.tag===30&&(t.flags&18874368)!==0&&t.stateNode.paired)){var e=t.memoizedProps;if(e.name==null||e.name==="auto")throw Error(J(544));var n=e.name;e=Ls(e.default,e.share),e!=="none"&&(Mo(t,n,e,null,!1)||ls(t.child,!1))}t=t.sibling}}function nm(t,e){if(t.tag===30){var n=t.stateNode,i=t.memoizedProps,s=Rs(i,n),r=Ls(i.default,n.paired?i.share:i.enter);r!=="none"?Mo(t,s,r,null,!1)?(Mu(t),n.paired||e||fo(t,i.onEnter)):ls(t.child,!1):Mu(t)}else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)nm(t,e),t=t.sibling;else Mu(t)}function im(t){if(jn!==null&&jn.size!==0){var e=jn;if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if(t.tag!==22||t.memoizedState===null){if(t.tag===30&&(t.flags&18874368)!==0){var n=t.memoizedProps,i=n.name;if(i!=null&&i!=="auto"){var s=e.get(i);if(s!==void 0){var r=Ls(n.default,n.share);if(r!=="none"&&(Mo(t,i,r,null,!1)?(r=t.stateNode,s.paired=r,r.paired=s,fo(t,n.onShare)):ls(t.child,!1)),e.delete(i),e.size===0)break}}}im(t)}t=t.sibling}}}function sm(t){if(t.tag===30){var e=t.memoizedProps,n=Rs(e,t.stateNode),i=jn!==null?jn.get(n):void 0,s=Ls(e.default,i!==void 0?e.share:e.exit);s!=="none"&&(Mo(t,n,s,null,!1)?i!==void 0?(s=t.stateNode,i.paired=s,s.paired=i,jn.delete(n),fo(t,e.onShare)):fo(t,e.onExit):ls(t.child,!1)),jn!==null&&im(t)}else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)sm(t),t=t.sibling;else jn!==null&&im(t)}function ES(t){for(t=t.child;t!==null;){if(t.tag===30){var e=t.memoizedProps,n=Rs(e,t.stateNode);e=Ls(e.default,e.update),t.flags&=-5,e!=="none"&&Mo(t,n,e,t.memoizedState=[],!1)}else(t.subtreeFlags&33554432)!==0&&ES(t);t=t.sibling}}function rm(t){if((t.subtreeFlags&18874368)!==0)for(t=t.child;t!==null;){if(t.tag!==22||t.memoizedState===null){if(t.tag===30&&(t.flags&18874368)!==0){var e=t.stateNode;e.paired!==null&&(e.paired=null,ls(t.child,!1))}rm(t)}t=t.sibling}}function Eu(t){if(t.tag===30)t.stateNode.paired=null,ls(t.child,!1),rm(t);else if((t.subtreeFlags&33554432)!==0)for(t=t.child;t!==null;)Eu(t),t=t.sibling;else rm(t)}function TS(t){for(t=t.child;t!==null;)t.tag===30?ls(t.child,!1):(t.subtreeFlags&33554432)!==0&&TS(t),t=t.sibling}function hg(t,e,n,i,s,r,a){for(var o=!1;e!==null;){if(e.tag===5){var l=e.stateNode;if(r!==null&&Vn<r.length){var c=r[Vn],h=Sm(l);(c.view||h.view)&&(o=!0);var p;if(p=(t.flags&4)===0)if(h.clip)p=!0;else{p=c.rect;var u=h.rect;p=p.y!==u.y||p.x!==u.x||p.height!==u.height||p.width!==u.width}p&&(t.flags|=4),h.abs?h=!c.abs:(c=c.rect,h=h.rect,h=c.height!==h.height||c.width!==h.width),h&&(t.flags|=32)}else t.flags|=32;(t.flags&4)!==0&&oA(l,Vn===0?n:n+"_"+Vn,s),o&&(t.flags&4)!==0||(ts===null&&(ts=[]),ts.push(l,Vn===0?i:i+"_"+Vn,e.memoizedProps)),Vn++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&a?t.flags|=e.flags&32:hg(t,e.child,n,i,s,r,a)&&(o=!0));e=e.sibling}return o}function bS(t,e){for(t=t.child;t!==null;){if(t.tag===30){var n=t.memoizedProps,i=t.stateNode,s=Rs(n,i),r=Ls(n.default,n.update);if(e){i=i.clones;var a=i===null?null:i.map(Cb)}else a=t.memoizedState,t.memoizedState=null;i=t;var o=t.child;Vn=0,s=hg(i,o,s,s,r,a,!1),(t.flags&4)!==0&&s&&(e||fo(t,n.onUpdate))}else(t.subtreeFlags&33554432)!==0&&bS(t,e);t=t.sibling}}var an=!1,lt=!1,Ji=!1,lp=!1,Kx=typeof WeakSet=="function"?WeakSet:Set,on=null,ji=!1,vl=!1,ju=!1,am=!1;function $T(t,e,n){if(t=t.containerInfo,xm=xo,t=f_(t),Fm(t)){if("selectionStart"in t)var i={start:t.selectionStart,end:t.selectionEnd};else e:{i=(i=t.ownerDocument)&&i.defaultView||window;var s=i.getSelection&&i.getSelection();if(s&&s.rangeCount!==0){i=s.anchorNode;var r=s.anchorOffset,a=s.focusNode;s=s.focusOffset;try{i.nodeType,a.nodeType}catch{i=null;break e}var o=0,l=-1,c=-1,h=0,p=0,u=t,d=null;t:for(;;){for(var x;u!==i||r!==0&&u.nodeType!==3||(l=o+r),u!==a||s!==0&&u.nodeType!==3||(c=o+s),u.nodeType===3&&(o+=u.nodeValue.length),(x=u.firstChild)!==null;)d=u,u=x;for(;;){if(u===t)break t;if(d===i&&++h===r&&(l=o),d===a&&++p===s&&(c=o),(x=u.nextSibling)!==null)break;u=d,d=u.parentNode}u=x}i=l===-1||c===-1?null:{start:l,end:c}}else i=null}i=i||{start:0,end:0}}else i=null;for(ym={focusedElem:t,selectionRange:i},xo=!1,n=(n&335544064)===n,on=e,e=n?9270:1024;on!==null;){if(t=on,n&&(i=t.deletions,i!==null))for(r=0;r<i.length;r++)n&&sm(i[r]);if(t.alternate===null&&(t.flags&2)!==0)n&&Qx(t),au(n);else{if(t.tag===22){if(i=t.alternate,t.memoizedState!==null){i!==null&&i.memoizedState===null&&n&&sm(i),au(n);continue}else if(i!==null&&i.memoizedState!==null){n&&Qx(t),au(n);continue}}i=t.child,(t.subtreeFlags&e)!==0&&i!==null?(i.return=t,on=i):(n&&ES(t),au(n))}}jn=null}function au(t){for(;on!==null;){var e=on,n=t,i=e.alternate,s=e.flags;switch(e.tag){case 0:case 11:case 15:break;case 1:if((s&1024)!==0&&i!==null){n=void 0,s=i.memoizedProps,i=i.memoizedState;var r=e.stateNode;try{var a=ca(e.type,s);n=r.getSnapshotBeforeUpdate(a,i),r.__reactInternalSnapshotBeforeUpdate=n}catch(o){dt(e,e.return,o)}}break;case 3:if((s&1024)!==0){if(i=e.stateNode.containerInfo,n=i.nodeType,n===9)Am(i);else if(n===1)switch(i.nodeName){case"HEAD":case"HTML":case"BODY":Am(i);break;default:i.textContent=""}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&i!==null&&(n=Rs(i.memoizedProps,i.stateNode),s=e.memoizedProps,s=Ls(s.default,s.update),s!=="none"&&Mo(i,n,s,i.memoizedState=[],!0));break;default:if((s&1024)!==0)throw Error(J(163))}if(i=e.sibling,i!==null){i.return=e.return,on=i;break}on=e.return}}function wS(t,e,n){var i=n.flags;switch(n.tag){case 0:case 11:case 15:$i(t,n),i&4&&tc(5,n);break;case 1:if($i(t,n),i&4)if(t=n.stateNode,e===null)try{t.componentDidMount()}catch(a){dt(n,n.return,a)}else{var s=ca(n.type,e.memoizedProps);e=e.memoizedState;try{t.componentDidUpdate(s,e,t.__reactInternalSnapshotBeforeUpdate)}catch(a){dt(n,n.return,a)}}i&64&&yS(n),i&512&&es(n,n.return);break;case 3:if($i(t,n),i&64&&(t=n.updateQueue,t!==null)){if(e=null,n.child!==null)switch(n.child.tag){case 27:case 5:e=n.child.stateNode;break;case 1:e=n.child.stateNode}try{R_(t,e)}catch(a){dt(n,n.return,a)}}break;case 27:e===null&&i&4&&AS(n);case 26:case 5:$i(t,n),e===null&&i&4&&em(n),i&512&&es(n,n.return);break;case 12:$i(t,n);break;case 31:$i(t,n),i&4&&US(t,n);break;case 13:$i(t,n),i&4&&IS(t,n),i&64&&(t=n.memoizedState,t!==null&&(t=t.dehydrated,t!==null&&(n=fb.bind(null,n),Wb(t,n))));break;case 22:if(i=n.memoizedState!==null||an,!i){var r=e!==null&&e.memoizedState!==null||lt;e=an,s=lt,an=i,(lt=r)&&!s?(i=2,(n.subtreeFlags&8772)!==0&&(i|=1),Bi(t,n,i)):$i(t,n),an=e,lt=s}break;case 30:$i(t,n),i&512&&es(n,n.return);break;case 7:i&512&&es(n,n.return);default:$i(t,n)}}function om(t,e){for(t=t.child;t!==null;)CS(t,e),t=t.sibling}function CS(t,e){switch(t.tag){case 5:case 26:try{var n=t.stateNode;if(e){var i=n.style;typeof i.setProperty=="function"?i.setProperty("display","none","important"):i.display="none"}else{var s=t.stateNode,r=t.memoizedProps.style,a=r!=null&&r.hasOwnProperty("display")?r.display:null;s.style.display=a==null||typeof a=="boolean"?"":(""+a).trim()}}catch(l){dt(t,t.return,l)}lm(t,e);break;case 6:try{t.stateNode.nodeValue=e?"":t.memoizedProps,nt=!0}catch(l){dt(t,t.return,l)}break;case 18:try{var o=t.stateNode;e?hy(o,!0):hy(t.stateNode,!1)}catch(l){dt(t,t.return,l)}break;case 22:case 23:t.memoizedState===null&&om(t,e);break;default:om(t,e)}}function lm(t,e){if(t.subtreeFlags&67108864)for(t=t.child;t!==null;){e:{var n=t,i=e;switch(n.tag){case 4:CS(n,i);break e;case 22:n.memoizedState===null&&lm(n,i);break e;default:lm(n,i)}}t=t.sibling}}function RS(t){var e=t.alternate;e!==null&&(t.alternate=null,RS(e)),t.child=null,t.deletions=null,t.sibling=null,t.tag===5&&(e=t.stateNode,e!==null&&cf(e)),t.stateNode=null,t.return=null,t.dependencies=null,t.memoizedProps=null,t.memoizedState=null,t.pendingProps=null,t.stateNode=null,t.updateQueue=null}var Rt=null,Hn=!1;function Ii(t,e,n){for(n=n.child;n!==null;)DS(t,e,n),n=n.sibling}function DS(t,e,n){if(ti&&typeof ti.onCommitFiberUnmount=="function")try{ti.onCommitFiberUnmount(Ql,n)}catch{}switch(n.tag){case 26:lt||mn(n,e),Ii(t,e,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!lt&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:lt||mn(n,e),wl(n);var i=Rt,s=Hn;Tr(n.type)&&(Rt=n.stateNode,Hn=!1),Ii(t,e,n),vA(n.stateNode,n.type,n.memoizedProps),Rt=i,Hn=s;break;case 5:lt||mn(n,e),wl(n);case 6:if(n.tag===6&&wl(n),i=Rt,s=Hn,Rt=null,Ii(t,e,n),Rt=i,Hn=s,Rt!==null)if(Hn)try{(Rt.nodeType===9?Rt.body:Rt.nodeName==="HTML"?Rt.ownerDocument.body:Rt).removeChild(n.stateNode),nt=!0}catch(r){dt(n,e,r)}else try{Rt.removeChild(n.stateNode),nt=!0}catch(r){dt(n,e,r)}break;case 18:Rt!==null&&(Hn?(t=Rt,fy(t.nodeType===9?t.body:t.nodeName==="HTML"?t.ownerDocument.body:t,n.stateNode),yo(t)):fy(Rt,n.stateNode));break;case 4:i=Rt,s=Hn,Rt=n.stateNode.containerInfo,Hn=!0,Ii(t,e,n),Rt=i,Hn=s;break;case 0:case 11:case 14:case 15:Ar(2,n,e),lt||Ar(4,n,e),Ii(t,e,n);break;case 1:lt||(mn(n,e),i=n.stateNode,typeof i.componentWillUnmount=="function"&&_S(n,e,i)),Ii(t,e,n);break;case 21:Ii(t,e,n);break;case 22:lt=(i=lt)||n.memoizedState!==null,Ii(t,e,n),lt=i;break;case 30:mn(n,e),Ii(t,e,n);break;case 7:lt||mn(n,e),Ii(t,e,n);break;default:Ii(t,e,n)}}function US(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null))){t=t.dehydrated;try{yo(t)}catch(n){dt(e,e.return,n)}}}function IS(t,e){if(e.memoizedState===null&&(t=e.alternate,t!==null&&(t=t.memoizedState,t!==null&&(t=t.dehydrated,t!==null))))try{yo(t)}catch(n){dt(e,e.return,n)}}function eb(t){switch(t.tag){case 31:case 13:case 19:var e=t.stateNode;return e===null&&(e=t.stateNode=new Kx),e;case 22:return t=t.stateNode,e=t._retryCache,e===null&&(e=t._retryCache=new Kx),e;default:throw Error(J(435,t.tag))}}function ou(t,e){var n=eb(t);e.forEach(function(i){if(!n.has(i)){n.add(i);var s=hb.bind(null,t,i);i.then(s,s)}})}function Bn(t,e,n){var i=e.deletions;if(i!==null)for(var s=0;s<i.length;s++){var r=i[s],a=t,o=e,l=o;e:for(;l!==null;){switch(l.tag){case 27:if(Tr(l.type)){Rt=l.stateNode,Hn=!1;break e}break;case 5:Rt=l.stateNode,Hn=!1;break e;case 3:case 4:Rt=l.stateNode.containerInfo,Hn=!0;break e}l=l.return}if(Rt===null)throw Error(J(160));DS(a,o,r),Rt=null,Hn=!1,a=r.alternate,a!==null&&(a.return=null),r.return=null}if(e.subtreeFlags&13886)for(e=e.child;e!==null;)BS(e,t,n),e=e.sibling}var Ni=null;function BS(t,e,n){var i=t.alternate,s=t.flags;switch(t.tag){case 0:case 11:case 14:case 15:if(s&4&&(i=t.updateQueue,i=i!==null?i.events:null,i!==null))for(var r=0;r<i.length;r++){var a=i[r];a.ref.impl=a.nextImpl}Bn(e,t,n),Nn(t),s&4&&(Ar(3,t,t.return),tc(3,t),Ar(5,t,t.return));break;case 1:Bn(e,t,n),Nn(t),s&512&&(lt||i===null||mn(i,i.return)),s&64&&an&&(t=t.updateQueue,t!==null&&(e=t.callbacks,e!==null&&(n=t.shared.hiddenCallbacks,t.shared.hiddenCallbacks=n===null?e:n.concat(e))));break;case 26:if(r=Ni,Bn(e,t,n),Nn(t),s&512&&(lt||i===null||mn(i,i.return)),s&4)if(s=i!==null?i.memoizedState:null,n=t.memoizedState,i===null)if(n===null)if(t.stateNode===null)if(an)t.stateNode=aA(t.type,t.memoizedProps,e.containerInfo,t);else{e:{e=t.type,n=t.memoizedProps,s=r.ownerDocument||r;t:switch(e){case"title":i=s.getElementsByTagName("title")[0],(!i||i[Jl]||i[gn]||i.namespaceURI==="http://www.w3.org/2000/svg"||i.hasAttribute("itemprop"))&&(i=s.createElement(e),s.head.insertBefore(i,s.querySelector("head > title"))),_n(i,e,n),i[gn]=t,ln(i),e=i;break e;case"link":if(r=_y("link","href",s).get(e+(n.href||""))){for(a=0;a<r.length;a++)if(i=r[a],i.getAttribute("href")===(n.href==null||n.href===""?null:n.href)&&i.getAttribute("rel")===(n.rel==null?null:n.rel)&&i.getAttribute("title")===(n.title==null?null:n.title)&&i.getAttribute("crossorigin")===(n.crossOrigin==null?null:n.crossOrigin)){r.splice(a,1);break t}}i=s.createElement(e),_n(i,e,n),s.head.appendChild(i);break;case"meta":if(r=_y("meta","content",s).get(e+(n.content||""))){for(a=0;a<r.length;a++)if(i=r[a],i.getAttribute("content")===(n.content==null?null:""+n.content)&&i.getAttribute("name")===(n.name==null?null:n.name)&&i.getAttribute("property")===(n.property==null?null:n.property)&&i.getAttribute("http-equiv")===(n.httpEquiv==null?null:n.httpEquiv)&&i.getAttribute("charset")===(n.charSet==null?null:n.charSet)){r.splice(a,1);break t}}i=s.createElement(e),_n(i,e,n),s.head.appendChild(i);break;default:throw Error(J(468,e))}i[gn]=t,ln(i),e=i}t.stateNode=e}else an||Tm(r,t.type,t.stateNode);else t.stateNode=yy(r,n,t.memoizedProps);else s!==n?(s===null?(e=i.stateNode,e===null||lt||e.parentNode.removeChild(e)):s.count--,n===null?an||Tm(r,t.type,t.stateNode):yy(r,n,t.memoizedProps)):n===null&&t.stateNode!==null&&ap(t,t.memoizedProps,i.memoizedProps);break;case 27:Bn(e,t,n),Nn(t),s&512&&(lt||i===null||mn(i,i.return)),i!==null&&s&4&&ap(t,t.memoizedProps,i.memoizedProps);break;case 5:if(r=Ji,Ji=!1,Bn(e,t,n),Ji=r,Nn(t),s&512&&(lt||i===null||mn(i,i.return)),t.flags&32){e=t.stateNode;try{lo(e,""),nt=!0}catch(h){dt(t,t.return,h)}}s&4&&t.stateNode!=null&&(e=t.memoizedProps,ap(t,e,i!==null?i.memoizedProps:e)),s&1024&&(lp=!0);break;case 6:if(Bn(e,t,n),Nn(t),s&4){if(t.stateNode===null)throw Error(J(162));e=t.memoizedProps,n=t.stateNode;try{n.nodeValue=e,nt=!0}catch(h){dt(t,t.return,h)}}break;case 3:if(nt=!1,Cu=null,r=Ni,Ni=kl(e.containerInfo),Bn(e,t,n),Ni=r,Nn(t),s&4&&i!==null&&i.memoizedState.isDehydrated)try{yo(e.containerInfo)}catch(h){dt(t,t.return,h)}lp&&(lp=!1,NS(t)),nt=!1;break;case 4:s=Ji,Ji=an,i=nx(),r=Ni,Ni=kl(t.stateNode.containerInfo),Bn(e,t,n),Nn(t),Ni=r,nt&&vl&&(ju=!0),nt=i,Ji=s;break;case 12:Bn(e,t,n),Nn(t);break;case 31:Bn(e,t,n),Nn(t),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 13:Bn(e,t,n),Nn(t),t.child.flags&8192&&t.memoizedState!==null!=(i!==null&&i.memoizedState!==null)&&(Af=ei()),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 22:r=t.memoizedState!==null,a=i!==null&&i.memoizedState!==null;var o=an,l=lt,c=Ji;an=o||r,Ji=c||r,lt=l||a,Bn(e,t,n),lt=l,Ji=c,an=o,Nn(t),s&8192&&(e=t.stateNode,e._visibility=r?e._visibility&-2:e._visibility|1,!r||i===null||a||an||lt||(e=a||lt,n=an,i=lt,an=r||an,lt=e,$s(t,2),an=n,lt=i),!r&&Ji||om(t,r)),s&4&&(e=t.updateQueue,e!==null&&(n=e.retryQueue,n!==null&&(e.retryQueue=null,ou(t,n))));break;case 19:Bn(e,t,n),Nn(t),s&4&&(e=t.updateQueue,e!==null&&(t.updateQueue=null,ou(t,e)));break;case 30:s&512&&(lt||i===null||mn(i,i.return)),s=nx(),r=vl,a=(n&335544064)===n,o=t.memoizedProps,vl=a&&Ls(o.default,o.update)!=="none",Bn(e,t,n),Nn(t),a&&i!==null&&nt&&(t.flags|=4),vl=r,nt=s;break;case 21:break;case 7:s&512&&(lt||i===null||mn(i,i.return)),i&&i.stateNode!==null&&(i.stateNode._fragmentFiber=t);default:Bn(e,t,n),Nn(t)}}function Nn(t){var e=t.flags;if(e&2){try{for(var n,i=t.return;i!==null;){if(SS(i)){n=i;break}i=i.return}i=null;for(var s=t.return;s!==null;){if(fg(s)){var r=s.stateNode;i===null?i=[r]:i.push(r)}if(ug(s))break;s=s.return}var a=i;if(n==null)throw Error(J(160));switch(n.tag){case 27:var o=n.stateNode,l=op(t);Ku(t,l,o,a);break;case 5:var c=n.stateNode;n.flags&32&&(lo(c,""),n.flags&=-33);var h=op(t);Ku(t,h,c,a);break;case 3:case 4:var p=n.stateNode.containerInfo,u=op(t);tm(t,u,p,a);break;default:throw Error(J(161))}}catch(d){dt(t,t.return,d)}t.flags&=-3}e&4096&&(t.flags&=-4097)}function NS(t){if(t.subtreeFlags&1024)for(t=t.child;t!==null;){var e=t;NS(e),e.tag===5&&e.flags&1024&&(e=e.stateNode,xo=!0,e.reset(),xo=!1),t=t.sibling}}function Na(t,e){if(e.subtreeFlags&9270)for(e=e.child;e!==null;)LS(e,t),e=e.sibling;else bS(e,!1)}function LS(t,e){var n=t.alternate;if(n===null)nm(t,!1);else switch(t.tag){case 3:if(am=ji=!1,Zx(),Na(e,t),!ji&&!ju){if(t=ts,t!==null)for(var i=0;i<t.length;i+=3){n=t[i];var s=t[i+1];lA(n,t[i+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:["none","none"]},{duration:0,fill:"forwards",pseudoElement:"::view-transition-group("+s+")"})}t=e.containerInfo,t=t.nodeType===9?t.documentElement:t.ownerDocument.documentElement,t!==null&&t.style.viewTransitionName===""&&(t.style.viewTransitionName="none",t.animate({opacity:[0,0],pointerEvents:["none","none"]},{duration:0,fill:"forwards",pseudoElement:"::view-transition-group(root)"}),t.animate({width:[0,0],height:[0,0]},{duration:0,fill:"forwards",pseudoElement:"::view-transition"})),am=!0}ts=null;break;case 5:Na(e,t);break;case 4:i=ji,ji=!1,Na(e,t),ji&&(ju=!0),ji=i;break;case 22:t.memoizedState===null&&(n.memoizedState!==null?nm(t,!1):Na(e,t));break;case 30:i=ji,s=Zx(),ji=!1,Na(e,t),ji&&(t.flags|=4);var r=t.memoizedProps,a=t.stateNode;e=Rs(r,a),a=Rs(n.memoizedProps,a);var o=Ls(r.default,r.update);o==="none"?e=!1:(r=n.memoizedState,n.memoizedState=null,n=t.child,Vn=0,e=hg(t,n,e,a,o,r,!0),Vn!==(r===null?0:r.length)&&(t.flags|=32)),(t.flags&4)!==0&&e?(fo(t,t.memoizedProps.onUpdate),ts=s):s!==null&&(s.push.apply(s,ts),ts=s),ji=(t.flags&32)!==0?!0:i;break;default:Na(e,t)}}function $i(t,e){if(e.subtreeFlags&8772)for(e=e.child;e!==null;)wS(t,e.alternate,e),e=e.sibling}function $s(t,e){for(t=t.child;t!==null;){var n=t,i=e;switch(n.tag){case 0:case 11:case 14:case 15:Ar(4,n,n.return),$s(n,i);break;case 1:mn(n,n.return);var s=n.stateNode;typeof s.componentWillUnmount=="function"&&_S(n,n.return,s),$s(n,i);break;case 27:(i&2)!==0&&vA(n.stateNode,n.type,n.memoizedProps);case 5:mn(n,n.return),n.tag!==5&&n.tag!==27||wl(n),$s(n,i);break;case 6:wl(n);break;case 26:mn(n,n.return),s=n.stateNode,n.memoizedState!==null||s===null||lt||s.parentNode.removeChild(s),$s(n,i);break;case 22:n.memoizedState===null&&$s(n,i);break;case 30:mn(n,n.return),$s(n,i);break;case 7:mn(n,n.return);default:$s(n,i)}t=t.sibling}}function Bi(t,e,n){for(n=(e.subtreeFlags&8772)!==0?n:n&-2,e=e.child;e!==null;){var i=e.alternate,s=t,r=e,a=r.flags,o=(n&1)!==0;switch(r.tag){case 0:case 11:case 15:Bi(s,r,n),tc(4,r);break;case 1:if(Bi(s,r,n),i=r,s=i.stateNode,typeof s.componentDidMount=="function")try{s.componentDidMount()}catch(h){dt(i,i.return,h)}if(i=r,s=i.updateQueue,s!==null){var l=i.stateNode;try{var c=s.shared.hiddenCallbacks;if(c!==null)for(s.shared.hiddenCallbacks=null,s=0;s<c.length;s++)C_(c[s],l)}catch(h){dt(i,i.return,h)}}o&&a&64&&yS(r),es(r,r.return);break;case 27:(n&2)!==0&&AS(r);case 5:r.tag!==5&&r.tag!==27||qx(r),Bi(s,r,n),o&&i===null&&a&4&&em(r),es(r,r.return);break;case 6:qx(r);break;case 26:l=r.stateNode,r.memoizedState!==null||l===null||an||Tm(kl(l.ownerDocument),r.type,l),Bi(s,r,n),o&&i===null&&a&4&&em(r),es(r,r.return);break;case 12:Bi(s,r,n);break;case 31:Bi(s,r,n),o&&a&4&&US(s,r);break;case 13:Bi(s,r,n),o&&a&4&&IS(s,r);break;case 22:r.memoizedState===null&&Bi(s,r,n),es(r,r.return);break;case 30:Bi(s,r,n),es(r,r.return);break;case 7:es(r,r.return);default:Bi(s,r,n)}e=e.sibling}}function dg(t,e){var n=null;t!==null&&t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(n=t.memoizedState.cachePool.pool),t=null,e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(t=e.memoizedState.cachePool.pool),t!==n&&(t!=null&&t.refCount++,n!=null&&$l(n))}function pg(t,e){t=null,e.alternate!==null&&(t=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==t&&(e.refCount++,t!=null&&$l(t))}function xi(t,e,n,i){var s=(n&335544064)===n;if(e.subtreeFlags&(s?10262:10256))for(e=e.child;e!==null;)PS(t,e,n,i),e=e.sibling;else s&&TS(e)}function PS(t,e,n,i){var s=(n&335544064)===n;s&&e.alternate===null&&e.return!==null&&e.return.alternate!==null&&Eu(e);var r=e.flags;switch(e.tag){case 0:case 11:case 15:xi(t,e,n,i),r&2048&&tc(9,e);break;case 1:xi(t,e,n,i);break;case 3:xi(t,e,n,i),s&&am&&(t=t.containerInfo,t=t.nodeType===9?t.body:t.nodeName==="HTML"?t.ownerDocument.body:t,t.style.viewTransitionName==="root"&&(t.style.viewTransitionName=""),t=t.ownerDocument.documentElement,t!==null&&t.style.viewTransitionName==="none"&&(t.style.viewTransitionName="")),r&2048&&(r=null,e.alternate!==null&&(r=e.alternate.memoizedState.cache),e=e.memoizedState.cache,e!==r&&(e.refCount++,r!=null&&$l(r)));break;case 12:if(r&2048){xi(t,e,n,i),r=e.stateNode;try{var a=e.memoizedProps,o=a.id,l=a.onPostCommit;typeof l=="function"&&l(o,e.alternate===null?"mount":"update",r.passiveEffectDuration,-0)}catch(c){dt(e,e.return,c)}}else xi(t,e,n,i);break;case 31:xi(t,e,n,i);break;case 13:xi(t,e,n,i);break;case 23:break;case 22:a=e.stateNode,o=e.alternate,e.memoizedState!==null?(s&&o!==null&&o.memoizedState===null&&Eu(o),a._visibility&2?xi(t,e,n,i):Cl(t,e)):(s&&o!==null&&o.memoizedState!==null&&Eu(e),a._visibility&2?xi(t,e,n,i):(a._visibility|=2,Pa(t,e,n,i,(e.subtreeFlags&10256)!==0||!1))),r&2048&&dg(o,e);break;case 24:xi(t,e,n,i),r&2048&&pg(e.alternate,e);break;case 30:s&&(r=e.alternate,r!==null&&(ls(r.child,!0),ls(e.child,!0))),xi(t,e,n,i);break;default:xi(t,e,n,i)}}function Pa(t,e,n,i,s){for(s=s&&((e.subtreeFlags&10256)!==0||!1),e=e.child;e!==null;){var r=t,a=e,o=n,l=i,c=a.flags;switch(a.tag){case 0:case 11:case 15:Pa(r,a,o,l,s),tc(8,a);break;case 23:break;case 22:var h=a.stateNode;a.memoizedState!==null?h._visibility&2?Pa(r,a,o,l,s):Cl(r,a):(h._visibility|=2,Pa(r,a,o,l,s)),s&&c&2048&&dg(a.alternate,a);break;case 24:Pa(r,a,o,l,s),s&&c&2048&&pg(a.alternate,a);break;default:Pa(r,a,o,l,s)}e=e.sibling}}function Cl(t,e){if(e.subtreeFlags&10256)for(e=e.child;e!==null;){var n=t,i=e,s=i.flags;switch(i.tag){case 22:Cl(n,i),s&2048&&dg(i.alternate,i);break;case 24:Cl(n,i),s&2048&&pg(i.alternate,i);break;default:Cl(n,i)}e=e.sibling}}var Zr=8192;function Yr(t,e,n){if(t.subtreeFlags&Zr)for(t=t.child;t!==null;)OS(t,e,n),t=t.sibling}function OS(t,e,n){switch(t.tag){case 26:Yr(t,e,n),t.flags&Zr&&(t.memoizedState!==null?sw(n,Ni,t.memoizedState,t.memoizedProps):(t=t.stateNode,(e&335544128)===e&&Ay(n,t)));break;case 5:Yr(t,e,n),t.flags&Zr&&(t=t.stateNode,(e&335544128)===e&&Ay(n,t));break;case 3:case 4:var i=Ni;Ni=kl(t.stateNode.containerInfo),Yr(t,e,n),Ni=i;break;case 22:t.memoizedState===null&&(i=t.alternate,i!==null&&i.memoizedState!==null?(i=Zr,Zr=16777216,Yr(t,e,n),Zr=i):Yr(t,e,n));break;case 30:if((t.flags&Zr)!==0&&(i=t.memoizedProps.name,i!=null&&i!=="auto")){var s=t.stateNode;s.paired=null,jn===null&&(jn=new Map),jn.set(i,s)}Yr(t,e,n);break;default:Yr(t,e,n)}}function FS(t){var e=t.alternate;if(e!==null&&(t=e.child,t!==null)){e.child=null;do e=t.sibling,t.sibling=null,t=e;while(t!==null)}}function ul(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];on=i,HS(i,t)}FS(t)}if(t.subtreeFlags&10256)for(t=t.child;t!==null;)zS(t),t=t.sibling}function zS(t){switch(t.tag){case 0:case 11:case 15:ul(t),t.flags&2048&&Ar(9,t,t.return);break;case 3:ul(t);break;case 12:ul(t);break;case 22:var e=t.stateNode;t.memoizedState!==null&&e._visibility&2&&(t.return===null||t.return.tag!==13)?(e._visibility&=-3,Tu(t)):ul(t);break;default:ul(t)}}function Tu(t){var e=t.deletions;if((t.flags&16)!==0){if(e!==null)for(var n=0;n<e.length;n++){var i=e[n];on=i,HS(i,t)}FS(t)}for(t=t.child;t!==null;){switch(e=t,e.tag){case 0:case 11:case 15:Ar(8,e,e.return),Tu(e);break;case 22:n=e.stateNode,n._visibility&2&&(n._visibility&=-3,Tu(e));break;default:Tu(e)}t=t.sibling}}function HS(t,e){for(;on!==null;){var n=on;switch(n.tag){case 0:case 11:case 15:Ar(8,n,e);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var i=n.memoizedState.cachePool.pool;i!=null&&i.refCount++}break;case 24:$l(n.memoizedState.cache)}if(i=n.child,i!==null)i.return=n,on=i;else e:for(n=t;on!==null;){i=on;var s=i.sibling,r=i.return;if(RS(i),i===n){on=null;break e}if(s!==null){s.return=r,on=s;break e}on=r}}}var tb={getCacheForType:function(t){var e=vn(Wt),n=e.data.get(t);return n===void 0&&(n=t(),e.data.set(t,n)),n},cacheSignal:function(){return vn(Wt).controller.signal}},nb=typeof WeakMap=="function"?WeakMap:Map,at=0,vt=null,Xe=null,Je=0,ft=0,Zn=null,ar=!1,Eo=!1,mg=!1,Bs=0,Ot=0,Mr=0,na=0,$u=0,$n=0,uo=0,Rl=null,Gn=null,cm=!1,Af=0,GS=0,ef=1/0,tf=null,mr=null,Nt=0,Pi=null,ua=null,os=0,um=0,fm=null,VS=null,io=null,so=null,ro=null,Dl=0,bu=null;function ii(){return(at&2)!==0&&Je!==0?Je&-Je:Ue.T!==null?vg():Yy()}function kS(){if($n===0)if((Je&536870912)===0||ke){var t=Qc;Qc<<=1,(Qc&3932160)===0&&(Qc=262144),$n=t}else $n=536870912;return t=Sn.current,t!==null&&(t.flags|=32),$n}function fo(t,e){if(e!=null){var n=t.stateNode,i=n.ref;i===null&&(i=n.ref=uA(Rs(t.memoizedProps,n))),so===null&&(so=[]),so.push(e.bind(null,i))}}function Wn(t,e,n){(t===vt&&(ft===2||ft===9)||t.cancelPendingCommit!==null)&&(ho(t,0),or(t,Je,$n,!1)),Kl(t,n),((at&2)===0||t!==vt)&&(t===vt&&((at&2)===0&&(na|=n),Ot===4&&or(t,Je,$n,!1)),us(t))}function WS(t,e,n){if((at&6)!==0)throw Error(J(327));var i=!n&&(e&127)===0&&(e&t.expiredLanes)===0||Zl(t,e),s=i?rb(t,e):cp(t,e,!0),r=i;do{if(s===0){Eo&&!i&&or(t,e,0,!1);break}else{if(n=t.current.alternate,r&&!ib(n)){s=cp(t,e,!1),r=!1;continue}if(s===2){if(r=e,t.errorRecoveryDisabledLanes&r)var a=0;else a=t.pendingLanes&-536870913,a=a!==0?a:a&536870912?536870912:0;if(a!==0){e=a;e:{var o=t;s=Rl;var l=o.current.memoizedState.isDehydrated;if(l&&(ho(o,a).flags|=256),a=cp(o,a,!1),a!==2&&a!==6){if(mg&&!l){o.errorRecoveryDisabledLanes|=r,na|=r,s=4;break e}r=Gn,Gn=s,r!==null&&(Gn===null?Gn=r:Gn.push.apply(Gn,r))}s=a}if(r=!1,s!==2)continue}}if(s===1){ho(t,0),or(t,e,0,!0);break}e:{switch(i=t,r=s,r){case 0:case 1:throw Error(J(345));case 4:if((e&4194048)!==e&&(e&62914560)!==e)break;case 6:or(i,e,$n,!ar);break e;case 2:Gn=null;break;case 3:case 5:break;default:throw Error(J(329))}if((e&62914560)===e&&(s=Af+300-ei(),10<s)){if(or(i,e,$n,!ar),lf(i,0,!0)!==0)break e;os=e,i.timeoutHandle=yg(Jx.bind(null,i,n,Gn,tf,cm,e,$n,na,uo,ar,r,"Throttled",-0,0),s);break e}Jx(i,n,Gn,tf,cm,e,$n,na,uo,ar,r,null,-0,0)}}break}while(!0);us(t)}function Jx(t,e,n,i,s,r,a,o,l,c,h,p,u,d){t.timeoutHandle=-1;var x=e.subtreeFlags,A=(r&335544064)===r;if(p=null,(A||x&8192||(x&16785408)===16785408)&&(p={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:is},jn=null,OS(e,r,p),A&&(x=p,A=t.containerInfo,A=(A.nodeType===9?A:A.ownerDocument).__reactViewTransition,A!=null&&(x.count++,x.waitingForViewTransition=!0,x=Wl.bind(x),A.finished.then(x,x))),x=(r&62914560)===r?Af-ei():(r&4194048)===r?GS-ei():0,x=rw(p,x),x!==null)){os=r,t.cancelPendingCommit=x($x.bind(null,t,e,r,n,i,s,a,o,l,c,h,p,null,u,d)),or(t,r,a,!c);return}$x(t,e,r,n,i,s,a,o,l,c,h,p)}function ib(t){for(var e=t;;){var n=e.tag;if((n===0||n===11||n===15)&&e.flags&16384&&(n=e.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var i=0;i<n.length;i++){var s=n[i],r=s.getSnapshot;s=s.value;try{if(!si(r(),s))return!1}catch{return!1}}if(n=e.child,e.subtreeFlags&16384&&n!==null)n.return=e,e=n;else{if(e===t)break;for(;e.sibling===null;){if(e.return===null||e.return===t)return!0;e=e.return}e.sibling.return=e.return,e=e.sibling}}return!0}function or(t,e,n,i){e=Gy(t,e),e&=~$u,e&=~na,t.suspendedLanes|=e,t.pingedLanes&=~e,i&&(t.warmLanes|=e),i=t.expirationTimes;for(var s=e;0<s;){var r=31-ni(s),a=1<<r;i[r]=-1,s&=~a}n!==0&&ky(t,n,e)}function Mf(){return(at&6)===0?(nc(0,!1),!1):!0}function gg(){if(Xe!==null){if(ft===0)var t=Xe.return;else t=Xe,Ts=ma=null,$m(t),eo=null,Ol=0,t=Xe;for(;t!==null;)xS(t.alternate,t),t=t.return;Xe=null}}function ho(t,e){var n=t.timeoutHandle;return n!==-1&&(t.timeoutHandle=-1,Tb(n)),n=t.cancelPendingCommit,n!==null&&(t.cancelPendingCommit=null,n()),os=0,gg(),vt=t,Xe=n=bs(t.current,null),Je=e,ft=0,Zn=null,ar=!1,Eo=Zl(t,e),mg=!1,uo=$n=$u=na=Mr=Ot=0,Gn=Rl=null,cm=!1,Bs=Gy(t,e),df(),n}function XS(t,e){ze=null,Ue.H=qu,e===Ao||e===gf?(e=Ex(),ft=3):e===Xm?(e=Ex(),ft=4):ft=e===og?8:e!==null&&typeof e=="object"&&typeof e.then=="function"?6:1,Zn=e,Xe===null&&(Ot=1,Qu(t,Mi(e,t.current)))}function YS(){var t=Sn.current;return t===null?!0:(Je&4194048)===Je?bn===null:(Je&62914560)===Je||(Je&536870912)!==0?t===bn:!1}function qS(){var t=Ue.H;return Ue.H=qu,t===null?qu:t}function QS(){var t=Ue.A;return Ue.A=tb,t}function nf(){Ot=4,ar||(Je&4194048)!==Je&&Sn.current!==null||(Eo=!0),(Mr&134217727)===0&&(na&134217727)===0||vt===null||or(vt,Je,$n,!1)}function cp(t,e,n){var i=at;at|=2;var s=qS(),r=QS();(vt!==t||Je!==e)&&(tf=null,ho(t,e)),e=!1;var a=Ot;e:do try{if(ft!==0&&Xe!==null){var o=Xe,l=Zn;switch(ft){case 8:gg(),a=6;break e;case 3:case 2:case 9:case 6:Sn.current===null&&(e=!0);var c=ft;if(ft=0,Zn=null,Za(t,o,l,c),n&&Eo){a=0;break e}break;default:c=ft,ft=0,Zn=null,Za(t,o,l,c)}}sb(),a=Ot;break}catch(h){XS(t,h)}while(!0);return e&&t.shellSuspendCounter++,Ts=ma=null,at=i,Ue.H=s,Ue.A=r,Xe===null&&(vt=null,Je=0,df()),a}function sb(){for(;Xe!==null;)ZS(Xe)}function rb(t,e){var n=at;at|=2;var i=qS(),s=QS();vt!==t||Je!==e?(tf=null,ef=ei()+500,ho(t,e)):Eo=Zl(t,e);e:do try{if(ft!==0&&Xe!==null){e=Xe;var r=Zn;t:switch(ft){case 1:ft=0,Zn=null,Za(t,e,r,1);break;case 2:case 9:if(Mx(r)){ft=0,Zn=null,jx(e);break}e=function(){ft!==2&&ft!==9||vt!==t||(ft=7),us(t)},r.then(e,e);break e;case 3:ft=7;break e;case 4:ft=5;break e;case 7:Mx(r)?(ft=0,Zn=null,jx(e)):(ft=0,Zn=null,Za(t,e,r,7));break;case 5:var a=null;switch(Xe.tag){case 26:a=Xe.memoizedState;case 5:case 27:var o=Xe;if(a?_A(a):o.stateNode.complete){ft=0,Zn=null;var l=o.sibling;if(l!==null)Xe=l;else{var c=o.return;c!==null?(Xe=c,Ef(c)):Xe=null}break t}}ft=0,Zn=null,Za(t,e,r,5);break;case 6:ft=0,Zn=null,Za(t,e,r,6);break;case 8:gg(),Ot=6;break e;default:throw Error(J(462))}}ab();break}catch(h){XS(t,h)}while(!0);return Ts=ma=null,Ue.H=i,Ue.A=s,at=n,Xe!==null?0:(vt=null,Je=0,df(),Ot)}function ab(){for(;Xe!==null&&!ME();)ZS(Xe)}function ZS(t){var e=vS(t.alternate,t,Bs);t.memoizedProps=t.pendingProps,e===null?Ef(t):Xe=e}function jx(t){var e=t,n=e.alternate;switch(e.tag){case 15:case 0:e=Hx(n,e,e.pendingProps,e.type,void 0,Je);break;case 11:e=Hx(n,e,e.pendingProps,e.type.render,e.ref,Je);break;case 5:$m(e);var i=e;i===cn&&(ke?(Hu(i),i.tag===5&&i.stateNode!=null&&(At=i.stateNode)):(Hu(i),ke=!0));default:xS(n,e),e=Xe=y_(e,Bs),e=vS(n,e,Bs)}t.memoizedProps=t.pendingProps,e===null?Ef(t):Xe=e}function Za(t,e,n,i){Ts=ma=null,$m(e),eo=null,Ol=0;var s=e.return;try{if(qT(t,s,e,n,Je)){Ot=1,Qu(t,Mi(n,t.current)),Xe=null;return}}catch(r){if(s!==null)throw Xe=s,r;Ot=1,Qu(t,Mi(n,t.current)),Xe=null;return}e.flags&32768?(ke||i===1?t=!0:Eo||(Je&536870912)!==0?t=!1:(ar=t=!0,(i===2||i===9||i===3||i===6)&&(i=Sn.current,i!==null&&i.tag===13&&(i.flags|=16384))),KS(e,t)):Ef(e)}function Ef(t){var e=t;do{if((e.flags&32768)!==0){KS(e,ar);return}t=e.return;var n=JT(e.alternate,e,Bs);if(n!==null){Xe=n;return}if(e=e.sibling,e!==null){Xe=e;return}Xe=e=t}while(e!==null);Ot===0&&(Ot=5)}function KS(t,e){do{var n=jT(t.alternate,t);if(n!==null){n.flags&=32767,Xe=n;return}if(n=t.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!e&&(t=t.sibling,t!==null)){Xe=t;return}Xe=t=n}while(t!==null);Ot=6,Xe=null}function $x(t,e,n,i,s,r,a,o,l,c,h,p){t.cancelPendingCommit=null;do Tf();while(Nt!==0);if((at&6)!==0)throw Error(J(327));if(e!==null){if(e===t.current)throw Error(J(177));t===vt&&(Xe=vt=null,Je=0),ua=e,Pi=t,os=n,fm=s,VS=i,ob(t,e,n,a,o,l,p)}}function ob(t,e,n,i,s,r,a){var o=e.lanes|e.childLanes;if(um=o,o|=zm,BE(t,n,o,i,s,r),so=null,(n&335544064)===n?(ro=LT(t),i=10262):(ro=null,i=10256),(e.subtreeFlags&i)!==0||(e.flags&i)!==0?(t.callbackNode=null,t.callbackPriority=0,db(Lu,function(){return mm(),null})):(t.callbackNode=null,t.callbackPriority=0),Ju=!1,i=(e.flags&13878)!==0,(e.subtreeFlags&13878)!==0||i){i=Ue.T,Ue.T=null,s=ot.p,ot.p=2,r=at,at|=4;try{$T(t,e,n)}finally{at=r,ot.p=s,Ue.T=i}}Nt=1,Ju?io=Ub(a,t.containerInfo,ro,hm,dm,cb,pm,mm,lb,null,null):(hm(),dm(),pm())}function lb(t){if(Nt!==0){var e=Pi.onRecoverableError;e(t,{componentStack:null})}}function cb(){Nt===3&&(Nt=0,LS(ua,Pi),Nt=4)}function hm(){if(Nt===1){Nt=0;var t=Pi,e=ua,n=os,i=(e.flags&13878)!==0;if((e.subtreeFlags&13878)!==0||i){i=Ue.T,Ue.T=null;var s=ot.p;ot.p=2;var r=at;at|=4;try{vl=ju=!1,BS(e,t,n),n=ym;var a=f_(t.containerInfo),o=n.focusedElem,l=n.selectionRange;if(a!==o&&o&&o.ownerDocument&&u_(o.ownerDocument.documentElement,o)){if(l!==null&&Fm(o)){var c=l.start,h=l.end;if(h===void 0&&(h=c),"selectionStart"in o)o.selectionStart=c,o.selectionEnd=Math.min(h,o.value.length);else{var p=o.ownerDocument||document,u=p&&p.defaultView||window;if(u.getSelection){var d=u.getSelection(),x=o.textContent.length,A=Math.min(l.start,x),g=l.end===void 0?A:Math.min(l.end,x);!d.extend&&A>g&&(a=g,g=A,A=a);var f=mx(o,A),m=mx(o,g);if(f&&m&&(d.rangeCount!==1||d.anchorNode!==f.node||d.anchorOffset!==f.offset||d.focusNode!==m.node||d.focusOffset!==m.offset)){var y=p.createRange();y.setStart(f.node,f.offset),d.removeAllRanges(),A>g?(d.addRange(y),d.extend(m.node,m.offset)):(y.setEnd(m.node,m.offset),d.addRange(y))}}}}for(p=[],d=o;d=d.parentNode;)d.nodeType===1&&p.push({element:d,left:d.scrollLeft,top:d.scrollTop});for(typeof o.focus=="function"&&o.focus(),o=0;o<p.length;o++){var _=p[o];_.element.scrollLeft=_.left,_.element.scrollTop=_.top}}xo=!!xm,ym=xm=null}finally{at=r,ot.p=s,Ue.T=i}}t.current=e,Nt=2}}function dm(){if(Nt===2){Nt=0;var t=Pi,e=ua,n=(e.flags&8772)!==0;if((e.subtreeFlags&8772)!==0||n){n=Ue.T,Ue.T=null;var i=ot.p;ot.p=2;var s=at;at|=4;try{wS(t,e.alternate,e)}finally{at=s,ot.p=i,Ue.T=n}}Nt=3}}function pm(){if(Nt===4||Nt===3){Nt=0;var t=io;io=null,EE();var e=Pi,n=ua,i=os,s=VS,r=(i&335544064)===i?10262:10256;if((n.subtreeFlags&r)!==0||(n.flags&r)!==0?Nt=5:(Nt=0,ua=Pi=null,JS(e,e.pendingLanes)),r=e.pendingLanes,r===0&&(mr=null),Im(i),n=n.stateNode,ti&&typeof ti.onCommitFiberRoot=="function")try{ti.onCommitFiberRoot(Ql,n,void 0,(n.current.flags&128)===128)}catch{}if(s!==null){n=Ue.T,r=ot.p,ot.p=2,Ue.T=null;try{for(var a=e.onRecoverableError,o=0;o<s.length;o++){var l=s[o];a(l.value,{componentStack:l.stack})}}finally{Ue.T=n,ot.p=r}}if(s=so,a=ro,ro=null,s!==null&&(so=null,a===null&&(a=[]),t!==null))for(l=0;l<s.length;l++)n=(0,s[l])(a),n!==void 0&&t.finished.finally(n);(os&3)!==0&&Tf(),us(e),r=e.pendingLanes,(i&261930)!==0&&(r&42)!==0?e===bu?Dl++:(Dl=0,bu=e):(Dl=0,bu=null),nc(0,!1)}}function JS(t,e){(t.pooledCacheLanes&=e)===0&&(e=t.pooledCache,e!=null&&(t.pooledCache=null,$l(e)))}function Tf(){return io!==null&&(io.skipTransition(),io=null),hm(),dm(),pm(),mm()}function mm(){if(Nt!==5)return!1;var t=Pi,e=um;um=0;var n=Im(os),i=Ue.T,s=ot.p;try{ot.p=32>n?32:n,Ue.T=null,n=fm,fm=null;var r=Pi,a=os;if(Nt=0,ua=Pi=null,os=0,(at&6)!==0)throw Error(J(331));var o=at;if(at|=4,zS(r.current),PS(r,r.current,a,n),at=o,nc(0,!1),ti&&typeof ti.onPostCommitFiberRoot=="function")try{ti.onPostCommitFiberRoot(Ql,r)}catch{}return!0}finally{ot.p=s,Ue.T=i,JS(t,e)}}function ey(t,e,n){e=Mi(n,e),e=Qp(t.stateNode,e,2),t=hr(t,e,2),t!==null&&(Kl(t,2),us(t))}function dt(t,e,n){if(t.tag===3)ey(t,t,n);else for(;e!==null;){if(e.tag===3){ey(e,t,n);break}else if(e.tag===1){var i=e.stateNode;if(typeof e.type.getDerivedStateFromError=="function"||typeof i.componentDidCatch=="function"&&(mr===null||!mr.has(i))){t=Mi(n,t),n=hS(2),i=hr(e,n,2),i!==null&&(dS(n,i,e,t),Kl(i,2),us(i));break}}e=e.return}}function up(t,e,n){var i=t.pingCache;if(i===null){i=t.pingCache=new nb;var s=new Set;i.set(e,s)}else s=i.get(e),s===void 0&&(s=new Set,i.set(e,s));s.has(n)||(mg=!0,s.add(n),t=ub.bind(null,t,e,n),e.then(t,t))}function ub(t,e,n){var i=t.pingCache;i!==null&&i.delete(e),t.pingedLanes|=t.suspendedLanes&n,t.warmLanes&=~n,vt===t&&(Je&n)===n&&((Ot===4||Ot===3&&(Je&62914560)===Je&&300>ei()-Af)&&(at&2)===0?ho(t,0):$u|=n,uo===Je&&(uo=0)),us(t)}function jS(t,e){e===0&&(e=Vy()),t=pa(t,e),t!==null&&(Kl(t,e),us(t))}function fb(t){var e=t.memoizedState,n=0;e!==null&&(n=e.retryLane),jS(t,n)}function hb(t,e){var n=0;switch(t.tag){case 31:case 13:var i=t.stateNode,s=t.memoizedState;s!==null&&(n=s.retryLane);break;case 19:i=t.stateNode;break;case 22:i=t.stateNode._retryCache;break;default:throw Error(J(314))}i!==null&&i.delete(e),jS(t,n)}function db(t,e){return Dm(t,e)}var po=null,Oa=null,gm=!1,sf=!1,fp=!1,lr=0;function us(t){t!==Oa&&t.next===null&&(Oa===null?po=Oa=t:Oa=Oa.next=t),sf=!0,gm||(gm=!0,mb())}function nc(t,e){if(!fp&&sf){fp=!0;do for(var n=!1,i=po;i!==null;){if(!e)if(t!==0){var s=i.pendingLanes;if(s===0)var r=0;else{var a=i.suspendedLanes,o=i.pingedLanes;r=(1<<31-ni(42|t)+1)-1,r&=s&~(a&~o),r=r&201326741?r&201326741|1:r?r|2:0}r!==0&&(n=!0,ty(i,r))}else r=Je,r=lf(i,i===vt?r:0,i.cancelPendingCommit!==null||i.timeoutHandle!==-1),(r&3)===0||Zl(i,r)||(n=!0,ty(i,r));i=i.next}while(n);fp=!1}}function pb(){$S()}function $S(){sf=gm=!1;var t=0;lr!==0&&Eb()&&(t=lr);for(var e=ei(),n=null,i=po;i!==null;){var s=i.next,r=eA(i,e);r===0?(i.next=null,n===null?po=s:n.next=s,s===null&&(Oa=n)):(n=i,(t!==0||(r&3)!==0)&&(sf=!0)),i=s}Nt!==0&&Nt!==5||nc(t,!1),lr!==0&&(lr=0)}function eA(t,e){for(var n=t.suspendedLanes,i=t.pingedLanes,s=t.expirationTimes,r=t.pendingLanes&-62914561;0<r;){var a=31-ni(r),o=1<<a,l=s[a];l===-1?((o&n)===0||(o&i)!==0)&&(s[a]=IE(o,e)):l<=e&&(t.expiredLanes|=o),r&=~o}if(e=vt,n=Je,n=lf(t,t===e?n:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i=t.callbackNode,n===0||t===e&&(ft===2||ft===9)||t.cancelPendingCommit!==null)return i!==null&&i!==null&&kd(i),t.callbackNode=null,t.callbackPriority=0;if((n&3)===0||Zl(t,n)){if(e=n&-n,e===t.callbackPriority)return e;switch(i!==null&&kd(i),Im(n)){case 2:case 8:n=zy;break;case 32:n=Lu;break;case 268435456:n=Hy;break;default:n=Lu}return i=tA.bind(null,t),n=Dm(n,i),t.callbackPriority=e,t.callbackNode=n,e}return i!==null&&i!==null&&kd(i),t.callbackPriority=2,t.callbackNode=null,2}function tA(t,e){if(Nt!==0&&Nt!==5)return t.callbackNode=null,t.callbackPriority=0,null;var n=t.callbackNode;if(Tf()&&t.callbackNode!==n)return null;var i=Je;return i=lf(t,t===vt?i:0,t.cancelPendingCommit!==null||t.timeoutHandle!==-1),i===0?null:(WS(t,i,e),eA(t,ei()),t.callbackNode!=null&&t.callbackNode===n?tA.bind(null,t):null)}function ty(t,e){if(Tf())return null;WS(t,e,!0)}function mb(){bb(function(){(at&6)!==0?Dm(Fy,pb):$S()})}function vg(){if(lr===0){var t=aa;t===0&&(t=qc,qc<<=1,(qc&261888)===0&&(qc=256)),lr=t}return lr}function ny(t){return t==null||typeof t=="symbol"||typeof t=="boolean"?null:typeof t=="function"?t:pu(t)}function gb(t,e,n,i,s){if(e==="submit"&&n&&n.stateNode===s){var r=ny((s[Yn]||null).action),a=i.submitter;a&&(e=(e=a[Yn]||null)?ny(e.formAction):a.getAttribute("formAction"),e!==null&&(r=e,a=null));var o=new uf("action","action",null,i,s);t.push({event:o,listeners:[{instance:null,listener:function(){if(i.defaultPrevented){if(lr!==0){var l=new FormData(s,a);Yp(n,{pending:!0,data:l,method:s.method,action:r},null,l)}}else typeof r=="function"&&(o.preventDefault(),l=new FormData(s,a),Yp(n,{pending:!0,data:l,method:s.method,action:r},r,l))},currentTarget:s}]})}}for(lu=0;lu<Lp.length;lu++)cu=Lp[lu],iy=cu.toLowerCase(),sy=cu[0].toUpperCase()+cu.slice(1),Oi(iy,"on"+sy);var cu,iy,sy,lu;Oi(d_,"onAnimationEnd");Oi(p_,"onAnimationIteration");Oi(m_,"onAnimationStart");Oi("dblclick","onDoubleClick");Oi("focusin","onFocus");Oi("focusout","onBlur");Oi(wT,"onTransitionRun");Oi(CT,"onTransitionStart");Oi(RT,"onTransitionCancel");Oi(g_,"onTransitionEnd");oo("onMouseEnter",["mouseout","mouseover"]);oo("onMouseLeave",["mouseout","mouseover"]);oo("onPointerEnter",["pointerout","pointerover"]);oo("onPointerLeave",["pointerout","pointerover"]);ha("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));ha("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));ha("onBeforeInput",["compositionend","keypress","textInput","paste"]);ha("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));ha("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));ha("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Hl="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),vb=new Set("beforetoggle cancel close invalid load scroll scrollend toggle".split(" ").concat(Hl));function nA(t,e){e=(e&4)!==0;for(var n=0;n<t.length;n++){var i=t[n],s=i.event;i=i.listeners;e:{var r=void 0;if(e)for(var a=i.length-1;0<=a;a--){var o=i[a],l=o.instance,c=o.currentTarget;if(o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){Ou(h)}s.currentTarget=null,r=l}else for(a=0;a<i.length;a++){if(o=i[a],l=o.instance,c=o.currentTarget,o=o.listener,l!==r&&s.isPropagationStopped())break e;r=o,s.currentTarget=c;try{r(s)}catch(h){Ou(h)}s.currentTarget=null,r=l}}}}function We(t,e){var n=e[j0];n===void 0&&(n=e[j0]=new Set);var i=t+"__bubble";n.has(i)||(iA(e,t,2,!1),n.add(i))}function hp(t,e,n){var i=0;e&&(i|=4),iA(n,t,i,e)}var uu="_reactListening"+Math.random().toString(36).slice(2);function xg(t){if(!t[uu]){t[uu]=!0,Qy.forEach(function(n){n!=="selectionchange"&&(vb.has(n)||hp(n,!1,t),hp(n,!0,t))});var e=t.nodeType===9?t:t.ownerDocument;e===null||e[uu]||(e[uu]=!0,hp("selectionchange",!1,e))}}function iA(t,e,n,i){switch(wA(e)){case 2:var s=cw;break;case 8:s=uw;break;default:s=Tg}n=s.bind(null,e,n,t),s=void 0,!Up||e!=="touchstart"&&e!=="touchmove"&&e!=="wheel"||(s=!0),i?s!==void 0?t.addEventListener(e,n,{capture:!0,passive:s}):t.addEventListener(e,n,!0):s!==void 0?t.addEventListener(e,n,{passive:s}):t.addEventListener(e,n,!1)}function dp(t,e,n,i,s){var r=i;if((e&1)===0&&(e&2)===0&&i!==null)e:for(;;){if(i===null)return;var a=i.tag;if(a===3||a===4){var o=i.stateNode.containerInfo;if(o===s)break;if(a===4)for(a=i.return;a!==null;){var l=a.tag;if((l===3||l===4)&&a.stateNode.containerInfo===s)return;a=a.return}for(;o!==null;){if(a=Kr(o),a===null)return;if(l=a.tag,l===5||l===6||l===26||l===27){i=r=a;continue e}o=o.parentNode}}i=i.return}n_(function(){var c=r,h=Nm(n),p=[];e:{var u=v_.get(t);if(u!==void 0){var d=uf,x=t;switch(t){case"keypress":if(gu(n)===0)break e;case"keydown":case"keyup":d=iT;break;case"focusin":x="focus",d=Zd;break;case"focusout":x="blur",d=Zd;break;case"beforeblur":case"afterblur":d=Zd;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":d=ax;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":d=XE;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":d=lT;break;case d_:case p_:case m_:d=QE;break;case g_:d=uT;break;case"scroll":case"scrollend":d=kE;break;case"wheel":d=hT;break;case"copy":case"cut":case"paste":d=KE;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":d=lx;break;case"submit":d=aT;break;case"toggle":case"beforetoggle":d=pT}var A=(e&4)!==0,g=!A&&(t==="scroll"||t==="scrollend"),f=A?u!==null?u+"Capture":null:u;A=[];for(var m=c,y;m!==null;){var _=m;if(y=_.stateNode,_=_.tag,_!==5&&_!==26&&_!==27||y===null||f===null||(_=Il(m,f),_!=null&&A.push(Gl(m,_,y))),g)break;m=m.return}0<A.length&&(u=new d(u,x,null,n,h),p.push({event:u,listeners:A}))}}if((e&7)===0){e:{if(d=t==="mouseover"||t==="pointerover",u=t==="mouseout"||t==="pointerout",d&&n!==Dp&&(x=n.relatedTarget||n.fromElement)&&(Kr(x)||x[_o]))break e;(u||d)&&(x=h.window===h?h:(d=h.ownerDocument)?d.defaultView||d.parentWindow:window,u?(d=n.relatedTarget||n.toElement,u=c,d=d?Kr(d):null,d!==null&&(g=ql(d),A=d.tag,d!==g||A!==5&&A!==27&&A!==6)&&(d=null)):(u=null,d=c),u!==d&&(A=ax,_="onMouseLeave",f="onMouseEnter",m="mouse",(t==="pointerout"||t==="pointerover")&&(A=lx,_="onPointerLeave",f="onPointerEnter",m="pointer"),g=u==null?x:ml(u),y=d==null?x:ml(d),x=new A(_,m+"leave",u,n,h),x.target=g,x.relatedTarget=y,_=null,Kr(h)===c&&(A=new A(f,m+"enter",d,n,h),A.target=y,A.relatedTarget=g,_=A),g=_,A=u&&d?xp(u,d,xb):null,u!==null&&ry(p,x,u,A,!1),d!==null&&g!==null&&ry(p,g,d,A,!0)))}e:{if(u=c?ml(c):window,d=u.nodeName&&u.nodeName.toLowerCase(),d==="select"||d==="input"&&u.type==="file")var T=hx;else if(fx(u))if(l_)T=ET;else{T=AT;var b=ST}else d=u.nodeName,!d||d.toLowerCase()!=="input"||u.type!=="checkbox"&&u.type!=="radio"?c&&Bm(c.elementType)&&(T=hx):T=MT;if(T&&(T=T(t,c))){o_(p,T,n,h);break e}b&&b(t,u,c)}switch(b=c?ml(c):window,t){case"focusin":(fx(b)||b.contentEditable==="true")&&(ka=b,Bp=c,_l=null);break;case"focusout":_l=Bp=ka=null;break;case"mousedown":Np=!0;break;case"contextmenu":case"mouseup":case"dragend":Np=!1,gx(p,n,h);break;case"selectionchange":if(bT)break;case"keydown":case"keyup":gx(p,n,h)}var v;if(Om)e:{switch(t){case"compositionstart":var E="onCompositionStart";break e;case"compositionend":E="onCompositionEnd";break e;case"compositionupdate":E="onCompositionUpdate";break e}E=void 0}else Va?r_(t,n)&&(E="onCompositionEnd"):t==="keydown"&&n.keyCode===229&&(E="onCompositionStart");E&&(s_&&n.locale!=="ko"&&(Va||E!=="onCompositionStart"?E==="onCompositionEnd"&&Va&&(v=i_()):(sr=h,Lm="value"in sr?sr.value:sr.textContent,Va=!0)),b=rf(c,E),0<b.length&&(E=new ox(E,t,null,n,h),p.push({event:E,listeners:b}),v?E.data=v:(v=a_(n),v!==null&&(E.data=v)))),(v=gT?vT(t,n):xT(t,n))&&(E=rf(c,"onBeforeInput"),0<E.length&&(b=new ox("onBeforeInput","beforeinput",null,n,h),p.push({event:b,listeners:E}),b.data=v)),gb(p,t,c,n,h)}nA(p,e)})}function Gl(t,e,n){return{instance:t,listener:e,currentTarget:n}}function rf(t,e){for(var n=e+"Capture",i=[];t!==null;){var s=t,r=s.stateNode;if(s=s.tag,s!==5&&s!==26&&s!==27||r===null||(s=Il(t,n),s!=null&&i.unshift(Gl(t,s,r)),s=Il(t,e),s!=null&&i.push(Gl(t,s,r))),t.tag===3)return i;t=t.return}return[]}function xb(t){if(t===null)return null;do t=t.return;while(t&&t.tag!==5&&t.tag!==27);return t||null}function ry(t,e,n,i,s){for(var r=e._reactName,a=[];n!==null&&n!==i;){var o=n,l=o.alternate,c=o.stateNode;if(o=o.tag,l!==null&&l===i)break;o!==5&&o!==26&&o!==27||c===null||(l=c,s?(c=Il(n,r),c!=null&&a.unshift(Gl(n,c,l))):s||(c=Il(n,r),c!=null&&a.push(Gl(n,c,l)))),n=n.return}a.length!==0&&t.push({event:e,listeners:a})}var yb=/\r\n?/g,_b=/\u0000|\uFFFD/g;function ay(t){return(typeof t=="string"?t:""+t).replace(yb,`
`).replace(_b,"")}function sA(t,e){return e=ay(e),ay(t)===e}function ht(t,e,n,i,s,r){switch(n){case"children":if(typeof i=="string")e==="body"||e==="textarea"&&i===""||lo(t,i);else if(typeof i=="number"||typeof i=="bigint")e!=="body"&&lo(t,""+i);else return;break;case"className":Kc(t,"class",i);break;case"tabIndex":Kc(t,"tabindex",i);break;case"dir":case"role":case"viewBox":case"width":case"height":Kc(t,n,i);break;case"style":t_(t,i,r);return;case"data":if(e!=="object"){Kc(t,"data",i);break}case"src":case"href":if(i===""&&(e!=="a"||n!=="href")){t.removeAttribute(n);break}if(i==null||typeof i=="function"||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=pu(i),t.setAttribute(n,i);break;case"action":case"formAction":if(typeof i=="function"){t.setAttribute(n,"javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')");break}else typeof r=="function"&&(n==="formAction"?(e!=="input"&&ht(t,e,"name",s.name,s,null),ht(t,e,"formEncType",s.formEncType,s,null),ht(t,e,"formMethod",s.formMethod,s,null),ht(t,e,"formTarget",s.formTarget,s,null)):(ht(t,e,"encType",s.encType,s,null),ht(t,e,"method",s.method,s,null),ht(t,e,"target",s.target,s,null)));if(i==null||typeof i=="symbol"||typeof i=="boolean"){t.removeAttribute(n);break}i=pu(i),t.setAttribute(n,i);break;case"onClick":i!=null&&(t.onclick=is);return;case"onScroll":i!=null&&We("scroll",t);return;case"onScrollEnd":i!=null&&We("scrollend",t);return;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));r?.__html!==n&&(t.innerHTML=n)}}break;case"multiple":t.multiple=i&&typeof i!="function"&&typeof i!="symbol";break;case"muted":t.muted=i&&typeof i!="function"&&typeof i!="symbol";break;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"defaultValue":case"defaultChecked":case"innerHTML":case"ref":break;case"autoFocus":break;case"xlinkHref":if(i==null||typeof i=="function"||typeof i=="boolean"||typeof i=="symbol"){t.removeAttribute("xlink:href");break}n=pu(i),t.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",n);break;case"contentEditable":case"spellCheck":case"draggable":case"value":case"autoReverse":case"externalResourcesRequired":case"focusable":case"preserveAlpha":i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,i):t.removeAttribute(n);break;case"inert":case"allowFullScreen":case"async":case"autoPlay":case"controls":case"credentialless":case"default":case"defer":case"disabled":case"disablePictureInPicture":case"disableRemotePlayback":case"formNoValidate":case"hidden":case"loop":case"noModule":case"noValidate":case"open":case"playsInline":case"readOnly":case"required":case"reversed":case"scoped":case"seamless":case"itemScope":i&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,""):t.removeAttribute(n);break;case"capture":case"download":i===!0?t.setAttribute(n,""):i!==!1&&i!=null&&typeof i!="function"&&typeof i!="symbol"?t.setAttribute(n,i):t.removeAttribute(n);break;case"cols":case"rows":case"size":case"span":i!=null&&typeof i!="function"&&typeof i!="symbol"&&!isNaN(i)&&1<=i?t.setAttribute(n,i):t.removeAttribute(n);break;case"rowSpan":case"start":i==null||typeof i=="function"||typeof i=="symbol"||isNaN(i)?t.removeAttribute(n):t.setAttribute(n,i);break;case"popover":We("beforetoggle",t),We("toggle",t),du(t,"popover",i);break;case"xlinkActuate":As(t,"http://www.w3.org/1999/xlink","xlink:actuate",i);break;case"xlinkArcrole":As(t,"http://www.w3.org/1999/xlink","xlink:arcrole",i);break;case"xlinkRole":As(t,"http://www.w3.org/1999/xlink","xlink:role",i);break;case"xlinkShow":As(t,"http://www.w3.org/1999/xlink","xlink:show",i);break;case"xlinkTitle":As(t,"http://www.w3.org/1999/xlink","xlink:title",i);break;case"xlinkType":As(t,"http://www.w3.org/1999/xlink","xlink:type",i);break;case"xmlBase":As(t,"http://www.w3.org/XML/1998/namespace","xml:base",i);break;case"xmlLang":As(t,"http://www.w3.org/XML/1998/namespace","xml:lang",i);break;case"xmlSpace":As(t,"http://www.w3.org/XML/1998/namespace","xml:space",i);break;case"is":du(t,"is",i);break;case"innerText":case"textContent":return;default:if(!(2<n.length)||n[0]!=="o"&&n[0]!=="O"||n[1]!=="n"&&n[1]!=="N")n=GE.get(n)||n,du(t,n,i);else return}nt=!0}function vm(t,e,n,i,s,r){switch(n){case"style":t_(t,i,r);return;case"dangerouslySetInnerHTML":if(i!=null){if(typeof i!="object"||!("__html"in i))throw Error(J(61));if(n=i.__html,n!=null){if(s.children!=null)throw Error(J(60));r?.__html!==n&&(t.innerHTML=n)}}break;case"children":if(typeof i=="string")lo(t,i);else if(typeof i=="number"||typeof i=="bigint")lo(t,""+i);else return;break;case"onScroll":i!=null&&We("scroll",t);return;case"onScrollEnd":i!=null&&We("scrollend",t);return;case"onClick":i!=null&&(t.onclick=is);return;case"suppressContentEditableWarning":case"suppressHydrationWarning":case"innerHTML":case"ref":return;case"innerText":case"textContent":return;default:if(!Zy.hasOwnProperty(n))e:{if(n[0]==="o"&&n[1]==="n"&&(s=n.endsWith("Capture"),r=n.slice(2,s?n.length-7:void 0),e=t[Yn]||null,e=e!=null?e[n]:null,typeof e=="function"&&t.removeEventListener(r,e,s),typeof i=="function")){typeof e!="function"&&e!==null&&(n in t?t[n]=null:t.hasAttribute(n)&&t.removeAttribute(n)),t.addEventListener(r,i,s);break e}nt=!0,n in t?t[n]=i:i===!0?t.setAttribute(n,""):du(t,n,i)}return}nt=!0}function _n(t,e,n){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"img":We("error",t),We("load",t);var i=!1,s=!1,r;for(r in n)if(n.hasOwnProperty(r)){var a=n[r];if(a!=null)switch(r){case"src":i=!0;break;case"srcSet":s=!0;break;case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:ht(t,e,r,a,n,null)}}s&&ht(t,e,"srcSet",n.srcSet,n,null),i&&ht(t,e,"src",n.src,n,null);return;case"input":We("invalid",t);var o=r=a=s=null,l=null,c=null;for(i in n)if(n.hasOwnProperty(i)){var h=n[i];if(h!=null)switch(i){case"name":s=h;break;case"type":a=h;break;case"checked":l=h;break;case"defaultChecked":c=h;break;case"value":r=h;break;case"defaultValue":o=h;break;case"children":case"dangerouslySetInnerHTML":if(h!=null)throw Error(J(137,e));break;default:ht(t,e,i,h,n,null)}}jy(t,r,o,l,c,a,s,!1);return;case"select":We("invalid",t),i=a=r=null;for(s in n)if(n.hasOwnProperty(s)&&(o=n[s],o!=null))switch(s){case"value":r=o;break;case"defaultValue":a=o;break;case"multiple":i=o;default:ht(t,e,s,o,n,null)}e=r,n=a,t.multiple=!!i,e!=null?Ja(t,!!i,e,!1):n!=null&&Ja(t,!!i,n,!0);return;case"textarea":We("invalid",t),r=s=i=null;for(a in n)if(n.hasOwnProperty(a)&&(o=n[a],o!=null))switch(a){case"value":i=o;break;case"defaultValue":s=o;break;case"children":r=o;break;case"dangerouslySetInnerHTML":if(o!=null)throw Error(J(91));break;default:ht(t,e,a,o,n,null)}e_(t,i,s,r);return;case"option":for(l in n)n.hasOwnProperty(l)&&(i=n[l],i!=null)&&(l==="selected"?t.selected=i&&typeof i!="function"&&typeof i!="symbol":ht(t,e,l,i,n,null));return;case"dialog":We("beforetoggle",t),We("toggle",t),We("cancel",t),We("close",t);break;case"iframe":case"object":We("load",t);break;case"video":case"audio":for(i=0;i<Hl.length;i++)We(Hl[i],t);break;case"image":We("error",t),We("load",t);break;case"details":We("toggle",t);break;case"embed":case"source":case"link":We("error",t),We("load",t);case"area":case"base":case"br":case"col":case"hr":case"keygen":case"meta":case"param":case"track":case"wbr":case"menuitem":for(c in n)if(n.hasOwnProperty(c)&&(i=n[c],i!=null))switch(c){case"children":case"dangerouslySetInnerHTML":throw Error(J(137,e));default:ht(t,e,c,i,n,null)}return;default:if(Bm(e)){for(h in n)n.hasOwnProperty(h)&&(i=n[h],i!==void 0&&vm(t,e,h,i,n,void 0));return}}for(o in n)n.hasOwnProperty(o)&&(i=n[o],i!=null&&ht(t,e,o,i,n,null))}var Sb={};function Ab(t,e,n,i){switch(e){case"div":case"span":case"svg":case"path":case"a":case"g":case"p":case"li":break;case"input":var s=null,r=null,a=null,o=null,l=null,c=null,h=null;for(d in n){var p=n[d];if(n.hasOwnProperty(d)&&p!=null)switch(d){case"checked":break;case"value":break;case"defaultValue":l=p;default:i.hasOwnProperty(d)||ht(t,e,d,null,i,p)}}for(var u in i){var d=i[u];if(p=n[u],i.hasOwnProperty(u)&&(d!=null||p!=null))switch(u){case"type":d!==p&&(nt=!0),r=d;break;case"name":d!==p&&(nt=!0),s=d;break;case"checked":d!==p&&(nt=!0),c=d;break;case"defaultChecked":d!==p&&(nt=!0),h=d;break;case"value":d!==p&&(nt=!0),a=d;break;case"defaultValue":d!==p&&(nt=!0),o=d;break;case"children":case"dangerouslySetInnerHTML":if(d!=null)throw Error(J(137,e));break;default:d!==p&&ht(t,e,u,d,i,p)}}Rp(t,a,o,l,c,h,r,s);return;case"select":d=a=o=u=null;for(r in n)if(l=n[r],n.hasOwnProperty(r)&&l!=null)switch(r){case"value":break;case"multiple":d=l;default:i.hasOwnProperty(r)||ht(t,e,r,null,i,l)}for(s in i)if(r=i[s],l=n[s],i.hasOwnProperty(s)&&(r!=null||l!=null))switch(s){case"value":r!==l&&(nt=!0),u=r;break;case"defaultValue":r!==l&&(nt=!0),o=r;break;case"multiple":r!==l&&(nt=!0),a=r;default:r!==l&&ht(t,e,s,r,i,l)}e=o,n=a,i=d,u!=null?Ja(t,!!n,u,!1):!!i!=!!n&&(e!=null?Ja(t,!!n,e,!0):Ja(t,!!n,n?[]:"",!1));return;case"textarea":d=u=null;for(o in n)if(s=n[o],n.hasOwnProperty(o)&&s!=null&&!i.hasOwnProperty(o))switch(o){case"value":break;case"children":break;default:ht(t,e,o,null,i,s)}for(a in i)if(s=i[a],r=n[a],i.hasOwnProperty(a)&&(s!=null||r!=null))switch(a){case"value":s!==r&&(nt=!0),u=s;break;case"defaultValue":s!==r&&(nt=!0),d=s;break;case"children":break;case"dangerouslySetInnerHTML":if(s!=null)throw Error(J(91));break;default:s!==r&&ht(t,e,a,s,i,r)}$y(t,u,d);return;case"option":for(var x in n)u=n[x],n.hasOwnProperty(x)&&u!=null&&!i.hasOwnProperty(x)&&(x==="selected"?t.selected=!1:ht(t,e,x,null,i,u));for(l in i)u=i[l],d=n[l],i.hasOwnProperty(l)&&u!==d&&(u!=null||d!=null)&&(l==="selected"?(u!==d&&(nt=!0),t.selected=u&&typeof u!="function"&&typeof u!="symbol"):ht(t,e,l,u,i,d));return;case"img":case"link":case"area":case"base":case"br":case"col":case"embed":case"hr":case"keygen":case"meta":case"param":case"source":case"track":case"wbr":case"menuitem":for(var A in n)u=n[A],n.hasOwnProperty(A)&&u!=null&&!i.hasOwnProperty(A)&&ht(t,e,A,null,i,u);for(c in i)if(u=i[c],d=n[c],i.hasOwnProperty(c)&&u!==d&&(u!=null||d!=null))switch(c){case"children":case"dangerouslySetInnerHTML":if(u!=null)throw Error(J(137,e));break;default:ht(t,e,c,u,i,d)}return;default:if(Bm(e)){for(var g in n)u=n[g],n.hasOwnProperty(g)&&u!==void 0&&!i.hasOwnProperty(g)&&vm(t,e,g,void 0,i,u);for(h in i)u=i[h],d=n[h],!i.hasOwnProperty(h)||u===d||u===void 0&&d===void 0||vm(t,e,h,u,i,d);return}}for(var f in n)u=n[f],n.hasOwnProperty(f)&&u!=null&&!i.hasOwnProperty(f)&&ht(t,e,f,null,i,u);for(p in i)u=i[p],d=n[p],!i.hasOwnProperty(p)||u===d||u==null&&d==null||ht(t,e,p,u,i,d)}function oy(t){switch(t){case"css":case"script":case"font":case"img":case"image":case"input":case"link":return!0;default:return!1}}function Mb(){if(typeof performance.getEntriesByType=="function"){for(var t=0,e=0,n=performance.getEntriesByType("resource"),i=0;i<n.length;i++){var s=n[i],r=s.transferSize,a=s.initiatorType,o=s.duration;if(r&&o&&oy(a)){for(a=0,o=s.responseEnd,i+=1;i<n.length;i++){var l=n[i],c=l.startTime;if(c>o)break;var h=l.transferSize,p=l.initiatorType;h&&oy(p)&&(l=l.responseEnd,a+=h*(l<o?1:(o-c)/(l-c)))}if(--i,e+=8*(r+a)/(s.duration/1e3),t++,10<t)break}}if(0<t)return e/t/1e6}return navigator.connection&&(t=navigator.connection.downlink,typeof t=="number")?t:5}var xm=null,ym=null;function Vl(t){return t.nodeType===9?t:t.ownerDocument}function ly(t){switch(t){case"http://www.w3.org/2000/svg":return 1;case"http://www.w3.org/1998/Math/MathML":return 2;default:return 0}}function rA(t,e){if(t===0)switch(e){case"svg":return 1;case"math":return 2;default:return 0}return t===1&&e==="foreignObject"?0:t}function aA(t,e,n,i){return n=Vl(n).createElement(t),n[gn]=i,n[Yn]=e,_n(n,t,e),ln(n),n}function _m(t,e){return t==="textarea"||t==="noscript"||typeof e.children=="string"||typeof e.children=="number"||typeof e.children=="bigint"||typeof e.dangerouslySetInnerHTML=="object"&&e.dangerouslySetInnerHTML!==null&&e.dangerouslySetInnerHTML.__html!=null}var pp=null;function Eb(){var t=window.event;return t&&t.type==="popstate"?t===pp?!1:(pp=t,!0):(pp=null,!1)}var yg=typeof setTimeout=="function"?setTimeout:void 0,Tb=typeof clearTimeout=="function"?clearTimeout:void 0,cy=typeof Promise=="function"?Promise:void 0,uy=typeof requestAnimationFrame=="function"?requestAnimationFrame:yg,bb=typeof queueMicrotask=="function"?queueMicrotask:typeof cy<"u"?function(t){return cy.resolve(null).then(t).catch(wb)}:yg;function wb(t){setTimeout(function(){throw t})}function Tr(t){return t==="head"}function fy(t,e){var n=e,i=0;do{var s=n.nextSibling;if(t.removeChild(n),s&&s.nodeType===8)if(n=s.data,n==="/$"||n==="/&"){if(i===0){t.removeChild(s),yo(e);return}i--}else if(n==="$"||n==="$?"||n==="$~"||n==="$!"||n==="&")i++;else if(n==="html")gp(t.ownerDocument.documentElement);else if(n==="head"){n=t.ownerDocument.head,gp(n);for(var r=n.firstChild;r;){var a=r.nextSibling,o=r.nodeName;r[Jl]||o==="SCRIPT"||o==="STYLE"||o==="LINK"&&r.rel.toLowerCase()==="stylesheet"||n.removeChild(r),r=a}}else n==="body"&&gp(t.ownerDocument.body);n=s}while(n);yo(e)}function hy(t,e){var n=t;t=0;do{var i=n.nextSibling;if(n.nodeType===1?e?(n._stashedDisplay=n.style.display,n.style.display="none"):(n.style.display=n._stashedDisplay||"",n.getAttribute("style")===""&&n.removeAttribute("style")):n.nodeType===3&&(e?(n._stashedText=n.nodeValue,n.nodeValue=""):n.nodeValue=n._stashedText||""),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(t===0)break;t--}else n!=="$"&&n!=="$?"&&n!=="$~"&&n!=="$!"||t++;n=i}while(n)}function oA(t,e,n){if(e=CSS.escape(e)!==e?"r-"+btoa(e).replace(/=/g,""):e,t.style.viewTransitionName=e,n!=null&&(t.style.viewTransitionClass=n),n=getComputedStyle(t),n.display==="inline"){if(e=t.getClientRects(),e.length===1)var i=1;else for(var s=i=0;s<e.length;s++){var r=e[s];0<r.width&&0<r.height&&i++}i===1&&(t=t.style,t.display=e.length===1?"inline-block":"block",t.marginTop="-"+n.paddingTop,t.marginBottom="-"+n.paddingBottom)}}function lA(t,e){t=t.style,e=e.style;var n=e!=null?e.hasOwnProperty("viewTransitionName")?e.viewTransitionName:e.hasOwnProperty("view-transition-name")?e["view-transition-name"]:null:null;t.viewTransitionName=n==null||typeof n=="boolean"?"":(""+n).trim(),n=e!=null?e.hasOwnProperty("viewTransitionClass")?e.viewTransitionClass:e.hasOwnProperty("view-transition-class")?e["view-transition-class"]:null:null,t.viewTransitionClass=n==null||typeof n=="boolean"?"":(""+n).trim(),t.display==="inline-block"&&(e==null?t.display=t.margin="":(n=e.display,t.display=n==null||typeof n=="boolean"?"":n,n=e.margin,n!=null?t.margin=n:(n=e.hasOwnProperty("marginTop")?e.marginTop:e["margin-top"],t.marginTop=n==null||typeof n=="boolean"?"":n,e=e.hasOwnProperty("marginBottom")?e.marginBottom:e["margin-bottom"],t.marginBottom=e==null||typeof e=="boolean"?"":e)))}function cA(t,e,n){return n=n.ownerDocument.defaultView,{rect:t,abs:e.position==="absolute"||e.position==="fixed",clip:e.clipPath!=="none"||e.overflow!=="visible"||e.filter!=="none"||e.mask!=="none"||e.mask!=="none"||e.borderRadius!=="0px",view:0<=t.bottom&&0<=t.right&&t.top<=n.innerHeight&&t.left<=n.innerWidth}}function Sm(t){var e=t.getBoundingClientRect(),n=getComputedStyle(t);return cA(e,n,t)}function Cb(t){var e=t.getBoundingClientRect();e=new DOMRect(e.x+2e4,e.y+2e4,e.width,e.height);var n=getComputedStyle(t);return cA(e,n,t)}function Rb(t){return t.documentElement.clientHeight}function Db(t){this.addEventListener("load",t),this.addEventListener("error",t)}function Ub(t,e,n,i,s,r,a,o,l){var c=e.nodeType===9?e:e.ownerDocument;try{var h=c.startViewTransition({update:function(){var u=c.defaultView,d=u.navigation&&u.navigation.transition,x=c.fonts.status;i();var A=[];if(x==="loaded"&&(Rb(c),c.fonts.status==="loading"&&A.push(c.fonts.ready)),x=A.length,t!==null)for(var g=t.suspenseyImages,f=0,m=0;m<g.length;m++){var y=g[m];if(!y.complete){var _=y.getBoundingClientRect();if(0<_.bottom&&0<_.right&&_.top<u.innerHeight&&_.left<u.innerWidth){if(f+=SA(y),f>Ru){A.length=x;break}y=new Promise(Db.bind(y)),A.push(y)}}}if(0<A.length)return u=Promise.race([Promise.all(A),new Promise(function(T){return setTimeout(T,500)})]).then(s,s),(d?Promise.allSettled([d.finished,u]):u).then(r,r);if(s(),d)return d.finished.then(r,r);r()},types:n});c.__reactViewTransition=h;var p=[];return h.ready.then(function(){for(var u=c.documentElement.getAnimations({subtree:!0}),d=0;d<u.length;d++){var x=u[d],A=x.effect,g=A.pseudoElement;if(g!=null&&g.startsWith("::view-transition")){p.push(x),x=A.getKeyframes();for(var f=g=void 0,m=!0,y=0;y<x.length;y++){var _=x[y],T=_.width;if(g===void 0)g=T;else if(g!==T){m=!1;break}if(T=_.height,f===void 0)f=T;else if(f!==T){m=!1;break}delete _.width,delete _.height,_.transform==="none"&&delete _.transform}m&&g!==void 0&&f!==void 0&&(A.setKeyframes(x),m=getComputedStyle(A.target,A.pseudoElement),m.width!==g||m.height!==f)&&(m=x[0],m.width=g,m.height=f,m=x[x.length-1],m.width=g,m.height=f,A.setKeyframes(x))}}a()},function(u){c.__reactViewTransition===h&&(c.__reactViewTransition=null);try{typeof u=="object"&&u!==null&&u.name==="InvalidStateError"&&(u.message==="View transition was skipped because document visibility state is hidden."||u.message==="Skipping view transition because document visibility state has become hidden."||u.message==="Skipping view transition because viewport size changed."||u.message==="Transition was aborted because of invalid state")&&(u=null),u!==null&&l(u)}finally{i(),s(),a()}}),h.finished.finally(function(){for(var u=0;u<p.length;u++)p[u].cancel();c.__reactViewTransition===h&&(c.__reactViewTransition=null),o()}),h}catch{return i(),s(),a(),null}}function Jr(t,e){this._scope=document.documentElement,this._selector="::view-transition-"+t+"("+e+")"}Jr.prototype.animate=function(t,e){return e=typeof e=="number"?{duration:e}:xt({},e),e.pseudoElement=this._selector,this._scope.animate(t,e)};Jr.prototype.getAnimations=function(){for(var t=this._scope,e=this._selector,n=t.getAnimations({subtree:!0}),i=[],s=0;s<n.length;s++){var r=n[s].effect;r!==null&&r.target===t&&r.pseudoElement===e&&i.push(n[s])}return i};Jr.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function uA(t){return{name:t,group:new Jr("group",t),imagePair:new Jr("image-pair",t),old:new Jr("old",t),new:new Jr("new",t)}}function ri(t){this._fragmentFiber=t,this._observers=this._eventListeners=null}ri.prototype.addEventListener=function(t,e,n){var i=null,s=null;if(!(n!=null&&typeof n!="boolean"&&(i=n.signal||null,i!==null&&i.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var r=this._eventListeners;if(fA(r,t,e,n)===-1){var a=this,o=e;n!=null&&typeof n!="boolean"&&n.once===!0&&(o=function(l){a.removeEventListener(t,e,n),typeof e=="function"?e.call(this,l):e.handleEvent(l)}),i!==null&&(s=a.removeEventListener.bind(a,t,e,n),i.addEventListener("abort",s,{once:!0}),s=i.removeEventListener.bind(i,"abort",s)),i=mo(n),r.push({type:t,listener:e,optionsOrUseCapture:n,attachedListener:o,cleanup:s}),Xn(this._fragmentFiber.child,!1,Ib,t,o,i)}this._eventListeners=r}};function Ib(t,e,n,i){return jt(t).addEventListener(e,n,i),!1}ri.prototype.removeEventListener=function(t,e,n){var i=this._eventListeners;if(i!==null&&(e=fA(i,t,e,n),e!==-1)){var s=i[e];n=s.attachedListener;var r=s.cleanup;s=mo(s.optionsOrUseCapture),Xn(this._fragmentFiber.child,!1,Bb,t,n,s),i.splice(e,1),r!==null&&r()}};function Bb(t,e,n,i){return jt(t).removeEventListener(e,n,i),!1}function mo(t){return t!=null&&typeof t!="boolean"&&(t.once===!0||t.signal instanceof AbortSignal)?{capture:t.capture,passive:t.passive}:t}function dy(t){return t==null?"c=0":typeof t=="boolean"?"c="+(t?"1":"0"):"c="+(t.capture?"1":"0")}function fA(t,e,n,i){if(t.length===0)return-1;i=dy(i);for(var s=0;s<t.length;s++){var r=t[s];if(r.type===e&&r.listener===n&&dy(r.optionsOrUseCapture)===i)return s}return-1}ri.prototype.dispatchEvent=function(t){var e=fa(this._fragmentFiber);if(e===null)return!0;e=jt(e);var n=this._eventListeners;if(n!==null&&0<n.length||!t.bubbles){var i=e.nodeType===9?e.createComment(""):document.createTextNode("");if(n)for(var s=0;s<n.length;s++){var r=n[s];i.addEventListener(r.type,r.attachedListener,mo(r.optionsOrUseCapture))}if(e.appendChild(i),t=i.dispatchEvent(t),n)for(s=0;s<n.length;s++)r=n[s],i.removeEventListener(r.type,r.attachedListener,mo(r.optionsOrUseCapture));return e.removeChild(i),t}return e.dispatchEvent(t)};ri.prototype.focus=function(t){Xn(this._fragmentFiber.child,!0,hA,t,void 0,void 0)};function hA(t,e){return t.tag===6?!1:(t=jt(t),Xb(t,e))}ri.prototype.focusLast=function(t){var e=[];Xn(this._fragmentFiber.child,!0,_g,e,void 0,void 0);for(var n=e.length-1;0<=n&&!hA(e[n],t);n--);};function _g(t,e){return e.push(t),!1}ri.prototype.blur=function(){var t=fa(this._fragmentFiber);t!==null&&(t=jt(t),t=Vl(t).activeElement,t!==null&&Xn(this._fragmentFiber.child,!1,Nb,t,void 0,void 0))};function Nb(t,e){return t.tag===6?!1:(t=jt(t),t===e||t.contains(e)?(e.blur(),!0):!1)}ri.prototype.observeUsing=function(t){this._observers===null&&(this._observers=new Set),this._observers.add(t),Xn(this._fragmentFiber.child,!1,Lb,t,void 0,void 0)};function Lb(t,e){return t.tag===6||(t=jt(t),e.observe(t)),!1}ri.prototype.unobserveUsing=function(t){var e=this._observers;if(e!==null&&e.has(t)){e.delete(t),Xn(this._fragmentFiber.child,!1,Pb,t,void 0,void 0);for(var n=e=0;n<Li.length;n++){var i=Li[n];i.fragmentInstance===this&&i.observer===t?t.unobserve(i.instance):Li[e++]=i}Li.length=e}};function Pb(t,e){return t.tag===6||(t=jt(t),e.unobserve(t)),!1}var Li=[],mp=!1;function Ob(t,e,n){Li.push({fragmentInstance:t,observer:e,instance:n}),mp||(mp=!0,Yb(function(){mp=!1;var i=Li;Li=[];for(var s=0;s<i.length;s++){var r=i[s];r.observer.unobserve(r.instance)}}))}ri.prototype.getClientRects=function(){var t=[];return Xn(this._fragmentFiber.child,!1,Fb,t,void 0,void 0),t};function Fb(t,e){if(t.tag===6){t=t.stateNode;var n=t.ownerDocument.createRange();n.selectNodeContents(t),e.push.apply(e,n.getClientRects())}else t=jt(t),e.push.apply(e,t.getClientRects());return!1}ri.prototype.getRootNode=function(t){var e=fa(this._fragmentFiber);return e===null?this:jt(e).getRootNode(t)};ri.prototype.compareDocumentPosition=function(t){var e=fa(this._fragmentFiber);if(e===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];Xn(this._fragmentFiber.child,!1,_g,n,void 0,void 0);var i=jt(e);if(n.length===0){if(n=i,Y0(this._fragmentFiber)){e:{for(e=this._fragmentFiber.return;e!==null;){if(e.tag===4){e=e.stateNode.containerInfo;break e}if(e.tag===3||e.tag===5||e.tag===27)break;e=e.return}e=null}e!=null&&(n=e)}e=this._fragmentFiber;var s=i=n.compareDocumentPosition(t);return n===t?s=Node.DOCUMENT_POSITION_CONTAINS:i&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=Ny(e)[1],n===null?s=Node.DOCUMENT_POSITION_PRECEDING:(t=jt(n).compareDocumentPosition(t),s=t===0||t&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),s|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}e=jt(n[0]),s=jt(n[n.length-1]);var r=Y0(this._fragmentFiber)?e.parentElement:i;if(r==null)return Node.DOCUMENT_POSITION_DISCONNECTED;i=r.compareDocumentPosition(e)&Node.DOCUMENT_POSITION_CONTAINED_BY,r=r.compareDocumentPosition(s)&Node.DOCUMENT_POSITION_CONTAINED_BY;var a=e.compareDocumentPosition(t),o=s.compareDocumentPosition(t),l=a&Node.DOCUMENT_POSITION_CONTAINED_BY||o&Node.DOCUMENT_POSITION_CONTAINED_BY;return o=i&&r&&a&Node.DOCUMENT_POSITION_FOLLOWING&&o&Node.DOCUMENT_POSITION_PRECEDING,e=i&&e===t||r&&s===t||l||o?Node.DOCUMENT_POSITION_CONTAINED_BY:!i&&e===t||!r&&s===t?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:a,e&Node.DOCUMENT_POSITION_DISCONNECTED||e&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||zb(e,this._fragmentFiber,n[0],n[n.length-1],t)?e:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function zb(t,e,n,i,s){var r=Kr(s);if(t&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!r)e:{for(;r!==null;){if(r.tag===7&&(r===e||r.alternate===e)){n=!0;break e}r=r.return}n=!1}return n}if(t&Node.DOCUMENT_POSITION_CONTAINS){if(r===null)return r=s.ownerDocument,s===r||s===r.documentElement||s===r.body;e:{for(r=e,e=fa(e);r!==null;){if(!(r.tag!==5&&r.tag!==3&&r.tag!==27||r!==e&&r.alternate!==e)){r=!0;break e}r=r.return}r=!1}return r}return t&Node.DOCUMENT_POSITION_PRECEDING?((e=!!r)&&!(e=r===n)&&(e=xp(n,r,q0),e===null?e=!1:(Xn(e,!0,mE,r,n),r=Fa,Fa=null,e=r!==null)),e):t&Node.DOCUMENT_POSITION_FOLLOWING?((e=!!r)&&!(e=r===i)&&(e=xp(i,r,q0),e===null?e=!1:(Xn(e,!0,gE,r,i),r=Fa,vp=Fa=null,e=r!==null)),e):!1}function py(t,e){var n=t.ownerDocument.createRange();n.selectNodeContents(t),t=n.getBoundingClientRect(),window.scrollTo(window.scrollX+t.left,e?window.scrollY+t.top:window.scrollY+t.bottom-window.innerHeight)}ri.prototype.scrollIntoView=function(t){if(typeof t=="object")throw Error(J(566));var e=[];Xn(this._fragmentFiber.child,!1,_g,e,void 0,void 0);var n=t!==!1;if(e.length===0){var i=Ny(this._fragmentFiber);if(i=n?i[1]||i[0]||fa(this._fragmentFiber):i[0]||i[1],i===null)return;if(i.tag===6){t=jt(i),py(t,n);return}if(i=jt(i),i.nodeType!==9){if(i.nodeType===11){n="host"in i?i.host:null,n!==null&&n.scrollIntoView(t);return}i.scrollIntoView(t)}}for(i=n?e.length-1:0;i!==(n?-1:e.length);){var s=e[i];s.tag===6?(s=jt(s),py(s,n)):jt(s).scrollIntoView(t),i+=n?-1:1}};function Hb(t,e){return t=jt(t),dA(t,e),!1}function dA(t,e){t.reactFragments==null&&(t.reactFragments=new Set),t.reactFragments.add(e)}function pA(t,e){var n=e._eventListeners;if(n!==null)for(var i=0;i<n.length;i++){var s=n[i];t.addEventListener(s.type,s.attachedListener,mo(s.optionsOrUseCapture))}t.nodeType!==3&&(n=e._observers,n!==null&&n.forEach(function(r){for(var a=0,o=0;o<Li.length;o++){var l=Li[o];(l.fragmentInstance!==e||l.observer!==r||l.instance!==t)&&(Li[a++]=l)}Li.length=a,r.observe(t)}),dA(t,e))}function Gb(t,e){var n=e._eventListeners;if(n!==null)for(var i=0;i<n.length;i++){var s=n[i];t.removeEventListener(s.type,s.attachedListener,mo(s.optionsOrUseCapture))}t.nodeType!==3&&(n=e._observers,n!==null&&n.forEach(function(r){typeof r.rootMargin=="string"?Ob(e,r,t):r.unobserve(t)}),t.reactFragments!=null&&t.reactFragments.delete(e))}function Am(t){var e=t.firstChild;for(e&&e.nodeType===10&&(e=e.nextSibling);e;){var n=e;switch(e=e.nextSibling,n.nodeName){case"HTML":case"HEAD":case"BODY":Am(n),cf(n);continue;case"SCRIPT":case"STYLE":continue;case"LINK":if(n.rel.toLowerCase()==="stylesheet")continue}t.removeChild(n)}}function Vb(t,e,n,i){for(;t.nodeType===1;){var s=n;if(t.nodeName.toLowerCase()!==e.toLowerCase()){if(!i&&(t.nodeName!=="INPUT"||t.type!=="hidden"))break}else if(i){if(!t[Jl])switch(e){case"meta":if(!t.hasAttribute("itemprop"))break;return t;case"link":if(r=t.getAttribute("rel"),r==="stylesheet"&&t.hasAttribute("data-precedence"))break;if(r!==s.rel||t.getAttribute("href")!==(s.href==null||s.href===""?null:s.href)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin)||t.getAttribute("title")!==(s.title==null?null:s.title))break;return t;case"style":if(t.hasAttribute("data-precedence"))break;return t;case"script":if(r=t.getAttribute("src"),(r!==(s.src==null?null:s.src)||t.getAttribute("type")!==(s.type==null?null:s.type)||t.getAttribute("crossorigin")!==(s.crossOrigin==null?null:s.crossOrigin))&&r&&t.hasAttribute("async")&&!t.hasAttribute("itemprop"))break;return t;default:return t}}else if(e==="input"&&t.type==="hidden"){var r=s.name==null?null:""+s.name;if(s.type==="hidden"&&t.getAttribute("name")===r)return t}else return t;if(t=Ti(t.nextSibling),t===null)break}return null}function kb(t,e,n){if(e==="")return null;for(;t.nodeType!==3;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!n||(t=Ti(t.nextSibling),t===null))return null;return t}function mA(t,e){for(;t.nodeType!==8;)if((t.nodeType!==1||t.nodeName!=="INPUT"||t.type!=="hidden")&&!e||(t=Ti(t.nextSibling),t===null))return null;return t}function Mm(t){return t.data==="$?"||t.data==="$~"}function Sg(t){return t.data==="$!"||t.data==="$?"&&t.ownerDocument.readyState!=="loading"}function Wb(t,e){var n=t.ownerDocument;if(t.data==="$~")t._reactRetry=e;else if(t.data!=="$?"||n.readyState!=="loading")e();else{var i=function(){e(),n.removeEventListener("DOMContentLoaded",i)};n.addEventListener("DOMContentLoaded",i),t._reactRetry=i}}function Ti(t){for(;t!=null;t=t.nextSibling){var e=t.nodeType;if(e===1||e===3)break;if(e===8){if(e=t.data,e==="$"||e==="$!"||e==="$?"||e==="$~"||e==="&"||e==="F!"||e==="F")break;if(e==="/$"||e==="/&")return null}}return t}var Em=null;function my(t){t=t.nextSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="/$"||n==="/&"){if(e===0)return Ti(t.nextSibling);e--}else n!=="$"&&n!=="$!"&&n!=="$?"&&n!=="$~"&&n!=="&"||e++}t=t.nextSibling}return null}function gy(t){t=t.previousSibling;for(var e=0;t;){if(t.nodeType===8){var n=t.data;if(n==="$"||n==="$!"||n==="$?"||n==="$~"||n==="&"){if(e===0)return t;e--}else n!=="/$"&&n!=="/&"||e++}t=t.previousSibling}return null}function Xb(t,e){function n(){i=!0}if(t.ownerDocument.activeElement===t)return!0;var i=!1;try{t.ownerDocument.addEventListener("focus",n,!0),(t.focus||HTMLElement.prototype.focus).call(t,e)}finally{t.ownerDocument.removeEventListener("focus",n,!0)}return i}function Yb(t){uy(function(){uy(function(e){return t(e)})})}function gA(t,e,n){switch(e=Vl(n),t){case"html":if(t=e.documentElement,!t)throw Error(J(452));return t;case"head":if(t=e.head,!t)throw Error(J(453));return t;case"body":if(t=e.body,!t)throw Error(J(454));return t;default:throw Error(J(451))}}function vA(t,e,n){for(var i in n){var s=n[i];n.hasOwnProperty(i)&&s!=null&&ht(t,e,i,null,Sb,s)}n.dangerouslySetInnerHTML!=null&&(t.textContent=""),t.onclick===is&&(t.onclick=null),cf(t)}function gp(t){for(var e=t.attributes;e.length;)t.removeAttributeNode(e[0]);cf(t)}var bi=new Map,vy=new Set;function kl(t){if(typeof t.getRootNode=="function"){var e=t.getRootNode();if(e.nodeType===9||e.nodeType===11)return e}return t.nodeType===9?t:t.ownerDocument}var Ps=ot.d;ot.d={f:qb,r:Qb,D:Zb,C:Kb,L:Jb,m:jb,X:ew,S:$b,M:tw};function qb(){var t=Ps.f(),e=Mf();return t||e}function Qb(t){var e=So(t);e!==null&&e.tag===5&&e.type==="form"?tS(e):Ps.r(t)}var To=typeof document>"u"?null:document;function xA(t,e,n){var i=To;if(i&&typeof e=="string"&&e){var s=Ai(e);s='link[rel="'+t+'"][href="'+s+'"]',typeof n=="string"&&(s+='[crossorigin="'+n+'"]'),vy.has(s)||(vy.add(s),t={rel:t,crossOrigin:n,href:e},i.querySelector(s)===null&&(e=i.createElement("link"),_n(e,"link",t),ln(e),i.head.appendChild(e)))}}function Zb(t){Ps.D(t),xA("dns-prefetch",t,null)}function Kb(t,e){Ps.C(t,e),xA("preconnect",t,e)}function Jb(t,e,n){Ps.L(t,e,n);var i=To;if(i&&t&&e){var s='link[rel="preload"][as="'+Ai(e)+'"]';e==="image"&&n&&n.imageSrcSet?(s+='[imagesrcset="'+Ai(n.imageSrcSet)+'"]',typeof n.imageSizes=="string"&&(s+='[imagesizes="'+Ai(n.imageSizes)+'"]')):s+='[href="'+Ai(t)+'"]';var r=s;switch(e){case"style":r=go(t);break;case"script":r=bo(t)}if(!(bi.has(r)||(t=xt({rel:"preload",href:e==="image"&&n&&n.imageSrcSet?void 0:t,as:e},n),bi.set(r,t),i.querySelector(s)!==null||e==="style"&&i.querySelector(ic(r))||e==="script"&&i.querySelector(sc(r))))){var a=i.createElement("link");_n(a,"link",t),e==="style"&&(a[Pu]=!0,a.onload=a.onerror=function(){qy(a)}),ln(a),i.head.appendChild(a)}}}function jb(t,e){Ps.m(t,e);var n=To;if(n&&t){var i=e&&typeof e.as=="string"?e.as:"script",s='link[rel="modulepreload"][as="'+Ai(i)+'"][href="'+Ai(t)+'"]',r=s;switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":r=bo(t)}if(!bi.has(r)&&(t=xt({rel:"modulepreload",href:t},e),bi.set(r,t),n.querySelector(s)===null)){switch(i){case"audioworklet":case"paintworklet":case"serviceworker":case"sharedworker":case"worker":case"script":if(n.querySelector(sc(r)))return}i=n.createElement("link"),_n(i,"link",t),ln(i),n.head.appendChild(i)}}}function $b(t,e,n){Ps.S(t,e,n);var i=To;if(i&&t){var s=Ka(i).hoistableStyles,r=go(t);e=e||"default";var a=s.get(r);if(!a){var o={loading:0,preload:null};if(a=i.querySelector(ic(r)))o.loading=5;else{t=xt({rel:"stylesheet",href:t,"data-precedence":e},n),(n=bi.get(r))&&Ag(t,n);var l=a=i.createElement("link");ln(l),_n(l,"link",t),l._p=new Promise(function(c,h){l.onload=c,l.onerror=h}),l.addEventListener("load",function(){o.loading|=1}),l.addEventListener("error",function(){o.loading|=2}),o.loading|=4,wu(a,e,i)}a={type:"stylesheet",instance:a,count:1,state:o},s.set(r,a)}}}function ew(t,e){Ps.X(t,e);var n=To;if(n&&t){var i=Ka(n).hoistableScripts,s=bo(t),r=i.get(s);r||(r=n.querySelector(sc(s)),r||(t=xt({src:t,async:!0},e),(e=bi.get(s))&&Mg(t,e),r=n.createElement("script"),ln(r),_n(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function tw(t,e){Ps.M(t,e);var n=To;if(n&&t){var i=Ka(n).hoistableScripts,s=bo(t),r=i.get(s);r||(r=n.querySelector(sc(s)),r||(t=xt({src:t,async:!0,type:"module"},e),(e=bi.get(s))&&Mg(t,e),r=n.createElement("script"),ln(r),_n(r,"link",t),n.head.appendChild(r)),r={type:"script",instance:r,count:1,state:null},i.set(s,r))}}function xy(t,e,n,i){var s=(s=cr.current)?kl(s):null;if(!s)throw Error(J(446));switch(t){case"meta":case"title":return null;case"style":return typeof n.precedence=="string"&&typeof n.href=="string"?(n=go(n.href),e=Ka(s).hoistableStyles,i=e.get(n),i||(i={type:"style",instance:null,count:0,state:null},e.set(n,i)),i):{type:"void",instance:null,count:0,state:null};case"link":if(n.rel==="stylesheet"&&typeof n.href=="string"&&typeof n.precedence=="string"){t=go(n.href);var r=Ka(s).hoistableStyles,a=r.get(t);if(a||(s=s.ownerDocument||s,a={type:"stylesheet",instance:null,count:0,state:{loading:0,preload:null}},r.set(t,a),(r=s.querySelector(ic(t)))?r._p||(a.instance=r,a.state.loading=5):(r=bi.get(t),r||(r={rel:"preload",as:"style",href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},bi.set(t,r)),nw(s,t,r,a.state))),e&&i===null)throw Error(J(528,""));return a}if(e&&i!==null)throw Error(J(529,""));return null;case"script":return e=n.async,n=n.src,typeof n=="string"&&e&&typeof e!="function"&&typeof e!="symbol"?(n=bo(n),e=Ka(s).hoistableScripts,i=e.get(n),i||(i={type:"script",instance:null,count:0,state:null},e.set(n,i)),i):{type:"void",instance:null,count:0,state:null};default:throw Error(J(444,t))}}function go(t){return'href="'+Ai(t)+'"'}function ic(t){return'link[rel="stylesheet"]['+t+"]"}function yA(t){return xt({},t,{"data-precedence":t.precedence,precedence:null})}function nw(t,e,n,i){if(e=t.querySelector('link[rel="preload"][as="style"]['+e+"]")){if(e[Pu]!==!0){i.loading=1;return}}else e=t.createElement("link"),e[Pu]=!0,e.onload=e.onerror=qy.bind(null,e),_n(e,"link",n),ln(e),t.head.appendChild(e);i.preload=e,e.addEventListener("load",function(){return i.loading|=1}),e.addEventListener("error",function(){return i.loading|=2})}function bo(t){return'[src="'+Ai(t)+'"]'}function sc(t){return"script[async]"+t}function yy(t,e,n){if(e.count++,e.instance===null)switch(e.type){case"style":var i=t.querySelector('style[data-href~="'+Ai(n.href)+'"]');if(i)return e.instance=i,ln(i),i;var s=xt({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return i=(t.ownerDocument||t).createElement("style"),ln(i),_n(i,"style",s),wu(i,n.precedence,t),e.instance=i;case"stylesheet":s=go(n.href);var r=t.querySelector(ic(s));if(r)return e.state.loading|=4,e.instance=r,ln(r),r;i=yA(n),(s=bi.get(s))&&Ag(i,s),r=(t.ownerDocument||t).createElement("link"),ln(r);var a=r;return a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),_n(r,"link",i),e.state.loading|=4,wu(r,n.precedence,t),e.instance=r;case"script":return r=bo(n.src),(s=t.querySelector(sc(r)))?(e.instance=s,ln(s),s):(i=n,(s=bi.get(r))&&(i=xt({},n),Mg(i,s)),t=t.ownerDocument||t,s=t.createElement("script"),ln(s),_n(s,"link",i),t.head.appendChild(s),e.instance=s);case"void":return null;default:throw Error(J(443,e.type))}else e.type==="stylesheet"&&(e.state.loading&4)===0&&(i=e.instance,e.state.loading|=4,wu(i,n.precedence,t));return e.instance}function wu(t,e,n){for(var i=n.querySelectorAll('link[rel="stylesheet"][data-precedence],style[data-precedence]'),s=i.length?i[i.length-1]:null,r=s,a=0;a<i.length;a++){var o=i[a];if(o.dataset.precedence===e)r=o;else if(r!==s)break}r?r.parentNode.insertBefore(t,r.nextSibling):(e=n.nodeType===9?n.head:n,e.insertBefore(t,e.firstChild))}function Ag(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.title==null&&(t.title=e.title)}function Mg(t,e){t.crossOrigin==null&&(t.crossOrigin=e.crossOrigin),t.referrerPolicy==null&&(t.referrerPolicy=e.referrerPolicy),t.integrity==null&&(t.integrity=e.integrity)}var Cu=null;function _y(t,e,n){if(Cu===null){var i=new Map,s=Cu=new Map;s.set(n,i)}else s=Cu,i=s.get(n),i||(i=new Map,s.set(n,i));if(i.has(t))return i;for(i.set(t,null),n=n.getElementsByTagName(t),s=0;s<n.length;s++){var r=n[s];if(!(r[Jl]||r[gn]||t==="link"&&r.getAttribute("rel")==="stylesheet")&&r.namespaceURI!=="http://www.w3.org/2000/svg"){var a=r.getAttribute(e)||"";a=t+a;var o=i.get(a);o?o.push(r):i.set(a,[r])}}return i}function Tm(t,e,n){t=t.ownerDocument||t,t.head.insertBefore(n,e==="title"?t.querySelector("head > title"):null)}function iw(t,e,n){if(n===1||e.itemProp!=null)return!1;switch(t){case"meta":case"title":return!0;case"style":if(typeof e.precedence!="string"||typeof e.href!="string"||e.href==="")break;return!0;case"link":if(typeof e.rel!="string"||typeof e.href!="string"||e.href===""||e.onLoad||e.onError)break;return e.rel==="stylesheet"?(t=e.disabled,typeof e.precedence=="string"&&t==null):!0;case"script":if(e.async&&typeof e.async!="function"&&typeof e.async!="symbol"&&!e.onLoad&&!e.onError&&e.src&&typeof e.src=="string")return!0}return!1}function Sy(t,e){return t==="img"&&e.src!=null&&e.src!==""&&e.onLoad==null&&e.loading!=="lazy"}function _A(t){return!(t.type==="stylesheet"&&(t.state.loading&3)===0)}function SA(t){return(t.width||100)*(t.height||100)*(typeof devicePixelRatio=="number"?devicePixelRatio:1)*.25}function Ay(t,e){typeof e.decode=="function"&&(t.imgCount++,e.complete||(t.imgBytes+=SA(e),t.suspenseyImages.push(e)),t=aw.bind(t),e.decode().then(t,t))}function sw(t,e,n,i){if(n.type==="stylesheet"&&(typeof i.media!="string"||matchMedia(i.media).matches!==!1)&&(n.state.loading&4)===0){if(n.instance===null){var s=go(i.href),r=e.querySelector(ic(s));if(r){e=r._p,e!==null&&typeof e=="object"&&typeof e.then=="function"&&(t.count++,t=Wl.bind(t),e.then(t,t)),n.state.loading|=4,n.instance=r,ln(r);return}r=e.ownerDocument||e,i=yA(i),(s=bi.get(s))&&Ag(i,s),r=r.createElement("link"),ln(r);var a=r;a._p=new Promise(function(o,l){a.onload=o,a.onerror=l}),_n(r,"link",i),n.instance=r}t.stylesheets===null&&(t.stylesheets=new Map),t.stylesheets.set(n,e),(e=n.state.preload)&&(n.state.loading&3)===0&&(t.count++,n=Wl.bind(t),e.addEventListener("load",n),e.addEventListener("error",n))}}var Ru=0;function rw(t,e){return t.stylesheets&&t.count===0&&Du(t,t.stylesheets),0<t.count||0<t.imgCount?function(n){var i=setTimeout(function(){if(t.stylesheets&&Du(t,t.stylesheets),t.unsuspend){var r=t.unsuspend;t.unsuspend=null,r()}},6e4+e);0<t.imgBytes&&Ru===0&&(Ru=62500*Mb());var s=setTimeout(function(){if(t.waitingForImages=!1,t.count===0&&(t.stylesheets&&Du(t,t.stylesheets),t.unsuspend)){var r=t.unsuspend;t.unsuspend=null,r()}},(t.imgBytes>Ru?50:800)+e);return t.unsuspend=n,function(){t.unsuspend=null,clearTimeout(i),clearTimeout(s)}}:null}function AA(t){if(t.count===0&&(t.imgCount===0||!t.waitingForImages)){if(t.stylesheets)Du(t,t.stylesheets);else if(t.unsuspend){var e=t.unsuspend;t.unsuspend=null,e()}}}function Wl(){this.count--,AA(this)}function aw(){this.imgCount--,AA(this)}var af=null;function Du(t,e){t.stylesheets=null,t.unsuspend!==null&&(t.count++,af=new Map,e.forEach(ow,t),af=null,Wl.call(t))}function ow(t,e){if(!(e.state.loading&4)){var n=af.get(t);if(n)var i=n.get(null);else{n=new Map,af.set(t,n);for(var s=t.querySelectorAll("link[data-precedence],style[data-precedence]"),r=0;r<s.length;r++){var a=s[r];(a.nodeName==="LINK"||a.getAttribute("media")!=="not all")&&(n.set(a.dataset.precedence,a),i=a)}i&&n.set(null,i)}s=e.instance,a=s.getAttribute("data-precedence"),r=n.get(a)||i,r===i&&n.set(null,s),n.set(a,s),this.count++,i=Wl.bind(this),s.addEventListener("load",i),s.addEventListener("error",i),r?r.parentNode.insertBefore(s,r.nextSibling):(t=t.nodeType===9?t.head:t,t.insertBefore(s,t.firstChild)),e.state.loading|=4}}var vo={$$typeof:ns,Provider:null,Consumer:null,_currentValue:jr,_currentValue2:jr,_threadCount:0};function lw(t,e,n,i,s,r,a,o,l){this.tag=1,this.containerInfo=t,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=Wd(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Wd(0),this.hiddenUpdates=Wd(null),this.identifierPrefix=i,this.onUncaughtError=s,this.onCaughtError=r,this.onRecoverableError=a,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=l,this.transitionTypes=null,this.incompleteTransitions=new Map}function MA(t,e,n,i,s,r,a,o,l,c,h,p){return t=new lw(t,e,n,a,l,c,h,p,o),e=1,r===!0&&(e|=24),r=kn(3,null,null,e),t.current=r,r.stateNode=t,e=km(),e.refCount++,t.pooledCache=e,e.refCount++,r.memoizedState={element:i,isDehydrated:n,cache:e},Ym(r),t}function EA(t){return t?(t=Ya,t):Ya}function TA(t,e,n,i,s,r){s=EA(s),i.context===null?i.context=s:i.pendingContext=s,i=fr(e),i.payload={element:n},r=r===void 0?null:r,r!==null&&(i.callback=r),n=hr(t,i,e),n!==null&&(Wn(n,t,e),Al(n,t,e))}function My(t,e){if(t=t.memoizedState,t!==null&&t.dehydrated!==null){var n=t.retryLane;t.retryLane=n!==0&&n<e?n:e}}function Eg(t,e){My(t,e),(t=t.alternate)&&My(t,e)}function bA(t){if(t.tag===13||t.tag===31){var e=pa(t,67108864);e!==null&&Wn(e,t,67108864),Eg(t,67108864)}}function Ey(t){if(t.tag===13||t.tag===31){var e=ii();e=Um(e);var n=pa(t,e);n!==null&&Wn(n,t,e),Eg(t,e)}}var xo=!0;function cw(t,e,n,i){var s=Ue.T;Ue.T=null;var r=ot.p;try{ot.p=2,Tg(t,e,n,i)}finally{ot.p=r,Ue.T=s}}function uw(t,e,n,i){var s=Ue.T;Ue.T=null;var r=ot.p;try{ot.p=8,Tg(t,e,n,i)}finally{ot.p=r,Ue.T=s}}function Tg(t,e,n,i){if(xo){var s=bm(i);if(s===null)dp(t,e,i,of,n),Ty(t,i);else if(hw(s,t,e,n,i))i.stopPropagation();else if(Ty(t,i),e&4&&-1<fw.indexOf(t)){for(;s!==null;){var r=So(s);if(r!==null)switch(r.tag){case 3:if(r=r.stateNode,r.current.memoizedState.isDehydrated){var a=qr(r.pendingLanes);if(a!==0){var o=r;for(o.pendingLanes|=2,o.entangledLanes|=2;a;){var l=1<<31-ni(a);o.entanglements[1]|=l,a&=~l}us(r),(at&6)===0&&(ef=ei()+500,nc(0,!1))}}break;case 31:case 13:o=pa(r,2),o!==null&&Wn(o,r,2),Mf(),Eg(r,2)}if(r=bm(i),r===null&&dp(t,e,i,of,n),r===s)break;s=r}s!==null&&i.stopPropagation()}else dp(t,e,i,null,n)}}function bm(t){return t=Nm(t),bg(t)}var of=null;function bg(t){if(of=null,t=Kr(t),t!==null){var e=ql(t);if(e===null)t=null;else{var n=e.tag;if(n===13){if(t=Uy(e),t!==null)return t;t=null}else if(n===31){if(t=Iy(e),t!==null)return t;t=null}else if(n===3){if(e.stateNode.current.memoizedState.isDehydrated)return e.tag===3?e.stateNode.containerInfo:null;t=null}else e!==t&&(t=null)}}return of=t,null}function wA(t){switch(t){case"beforetoggle":case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"seeked":case"submit":case"toggle":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"fullscreenerror":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 2;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"resize":case"scroll":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 8;case"message":switch(TE()){case Fy:return 2;case zy:return 8;case Lu:case bE:return 32;case Hy:return 268435456;default:return 32}default:return 32}}var wm=!1,gr=null,vr=null,xr=null,Xl=new Map,Yl=new Map,nr=[],fw="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset".split(" ");function Ty(t,e){switch(t){case"focusin":case"focusout":gr=null;break;case"dragenter":case"dragleave":vr=null;break;case"mouseover":case"mouseout":xr=null;break;case"pointerover":case"pointerout":Xl.delete(e.pointerId);break;case"gotpointercapture":case"lostpointercapture":Yl.delete(e.pointerId)}}function fl(t,e,n,i,s,r){return t===null||t.nativeEvent!==r?(t={blockedOn:e,domEventName:n,eventSystemFlags:i,nativeEvent:r,targetContainers:[s]},e!==null&&(e=So(e),e!==null&&bA(e)),t):(t.eventSystemFlags|=i,e=t.targetContainers,s!==null&&e.indexOf(s)===-1&&e.push(s),t)}function hw(t,e,n,i,s){switch(e){case"focusin":return gr=fl(gr,t,e,n,i,s),!0;case"dragenter":return vr=fl(vr,t,e,n,i,s),!0;case"mouseover":return xr=fl(xr,t,e,n,i,s),!0;case"pointerover":var r=s.pointerId;return Xl.set(r,fl(Xl.get(r)||null,t,e,n,i,s)),!0;case"gotpointercapture":return r=s.pointerId,Yl.set(r,fl(Yl.get(r)||null,t,e,n,i,s)),!0}return!1}function CA(t){var e=Kr(t.target);if(e!==null){var n=ql(e);if(n!==null){if(e=n.tag,e===13){if(e=Uy(n),e!==null){t.blockedOn=e,J0(t.priority,function(){Ey(n)});return}}else if(e===31){if(e=Iy(n),e!==null){t.blockedOn=e,J0(t.priority,function(){Ey(n)});return}}else if(e===3&&n.stateNode.current.memoizedState.isDehydrated){t.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}t.blockedOn=null}function Uu(t){if(t.blockedOn!==null)return!1;for(var e=t.targetContainers;0<e.length;){var n=bm(t.nativeEvent);if(n===null){n=t.nativeEvent;var i=new n.constructor(n.type,n);Dp=i,n.target.dispatchEvent(i),Dp=null}else return e=So(n),e!==null&&bA(e),t.blockedOn=n,!1;e.shift()}return!0}function by(t,e,n){Uu(t)&&n.delete(e)}function dw(){wm=!1,gr!==null&&Uu(gr)&&(gr=null),vr!==null&&Uu(vr)&&(vr=null),xr!==null&&Uu(xr)&&(xr=null),Xl.forEach(by),Yl.forEach(by)}function fu(t,e){t.blockedOn===e&&(t.blockedOn=null,wm||(wm=!0,$t.unstable_scheduleCallback($t.unstable_NormalPriority,dw)))}var hu=null;function wy(t){hu!==t&&(hu=t,$t.unstable_scheduleCallback($t.unstable_NormalPriority,function(){hu===t&&(hu=null);for(var e=0;e<t.length;e+=3){var n=t[e],i=t[e+1],s=t[e+2];if(typeof i!="function"){if(bg(i||n)===null)continue;break}var r=So(n);r!==null&&(t.splice(e,3),e-=3,Yp(r,{pending:!0,data:s,method:n.method,action:i},i,s))}}))}function yo(t){function e(l){return fu(l,t)}gr!==null&&fu(gr,t),vr!==null&&fu(vr,t),xr!==null&&fu(xr,t),Xl.forEach(e),Yl.forEach(e);for(var n=0;n<nr.length;n++){var i=nr[n];i.blockedOn===t&&(i.blockedOn=null)}for(;0<nr.length&&(n=nr[0],n.blockedOn===null);)CA(n),n.blockedOn===null&&nr.shift();if(n=(t.ownerDocument||t).$$reactFormReplay,n!=null)for(i=0;i<n.length;i+=3){var s=n[i],r=n[i+1],a=s[Yn]||null;if(typeof r=="function")a||wy(n);else if(a){var o=null;if(r&&r.hasAttribute("formAction")){if(s=r,a=r[Yn]||null)o=a.formAction;else if(bg(s)!==null)continue}else o=a.action;typeof o=="function"?n[i+1]=o:(n.splice(i,3),i-=3),wy(n)}}}function RA(){function t(r){r.canIntercept&&r.info==="react-transition"&&r.intercept({handler:function(){return new Promise(function(a){return s=a})},focusReset:"manual",scroll:"manual"})}function e(){s!==null&&(s(),s=null),i||setTimeout(n,20)}function n(){if(!i&&!navigation.transition){var r=navigation.currentEntry;r&&r.url!=null&&navigation.navigate(r.url,{state:r.getState(),info:"react-transition",history:"replace"})}}if(typeof navigation=="object"){var i=!1,s=null;return navigation.addEventListener("navigate",t),navigation.addEventListener("navigatesuccess",e),navigation.addEventListener("navigateerror",e),setTimeout(n,100),function(){i=!0,navigation.removeEventListener("navigate",t),navigation.removeEventListener("navigatesuccess",e),navigation.removeEventListener("navigateerror",e),s!==null&&(s(),s=null)}}}function wg(t){this._internalRoot=t}bf.prototype.render=wg.prototype.render=function(t){var e=this._internalRoot;if(e===null)throw Error(J(409));var n=e.current,i=ii();TA(n,i,t,e,null,null)};bf.prototype.unmount=wg.prototype.unmount=function(){var t=this._internalRoot;if(t!==null){this._internalRoot=null;var e=t.containerInfo;TA(t.current,2,null,t,null,null),Mf(),e[_o]=null}};function bf(t){this._internalRoot=t}bf.prototype.unstable_scheduleHydration=function(t){if(t){var e=Yy();t={blockedOn:null,target:t,priority:e};for(var n=0;n<nr.length&&e!==0&&e<nr[n].priority;n++);nr.splice(n,0,t),n===0&&CA(t)}};var Cy=Ry.version;if(Cy!=="19.3.1")throw Error(J(527,Cy,"19.3.1"));ot.findDOMNode=function(t){var e=t._reactInternals;if(e===void 0)throw typeof t.render=="function"?Error(J(188)):(t=Object.keys(t).join(","),Error(J(268,t)));return t=pE(e),t=t!==null?By(t):null,t=t===null?null:t.stateNode,t};var pw={bundleType:0,version:"19.3.1",rendererPackageName:"react-dom",currentDispatcherRef:Ue,reconcilerVersion:"19.3.1"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"&&(hl=__REACT_DEVTOOLS_GLOBAL_HOOK__,!hl.isDisabled&&hl.supportsFiber))try{Ql=hl.inject(pw),ti=hl}catch{}var hl;wf.createRoot=function(t,e){if(!Dy(t))throw Error(J(299));var n=!1,i="",s=cS,r=uS,a=fS;return e!=null&&(e.unstable_strictMode===!0&&(n=!0),e.identifierPrefix!==void 0&&(i=e.identifierPrefix),e.onUncaughtError!==void 0&&(s=e.onUncaughtError),e.onCaughtError!==void 0&&(r=e.onCaughtError),e.onRecoverableError!==void 0&&(a=e.onRecoverableError)),e=MA(t,1,!1,null,null,n,i,null,s,r,a,RA),t[_o]=e.current,xg(t),new wg(e)};wf.hydrateRoot=function(t,e,n){if(!Dy(t))throw Error(J(299));var i=!1,s="",r=cS,a=uS,o=fS,l=null;return n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(s=n.identifierPrefix),n.onUncaughtError!==void 0&&(r=n.onUncaughtError),n.onCaughtError!==void 0&&(a=n.onCaughtError),n.onRecoverableError!==void 0&&(o=n.onRecoverableError),n.formState!==void 0&&(l=n.formState)),e=MA(t,1,!0,e,n??null,i,s,l,r,a,o,RA),e.context=EA(null),n=e.current,i=ii(),i=Um(i),s=fr(i),s.callback=null,hr(n,s,i),n=i,e.current.lanes=n,Kl(e,n),us(e),t[_o]=e.current,xg(t),new bf(e)};wf.version="19.3.1"});var BA=Zi((ZU,IA)=>{"use strict";function UA(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(UA)}catch(t){console.error(t)}}UA(),IA.exports=DA()});var R1=Zi(_d=>{"use strict";var BU=Symbol.for("react.transitional.element"),NU=Symbol.for("react.fragment");function C1(t,e,n){var i=null;if(n!==void 0&&(i=""+n),e.key!==void 0&&(i=""+e.key),"key"in e){n={};for(var s in e)s!=="key"&&(n[s]=e[s])}else n=e;return e=n.ref,{$$typeof:BU,type:t,key:i,ref:e!==void 0?e:null,props:n}}_d.Fragment=NU;_d.jsx=C1;_d.jsxs=C1});var Fc=Zi((TP,D1)=>{"use strict";D1.exports=R1()});var Ad=Wr(nl()),B1=Wr(BA());var tM=0,tv=1,nM=2;var Ec=1,iM=2,Zo=3,ms=0,Qt=1,Fn=2,ki=0,Ko=1,nv=2,iv=3,sv=4,sM=5;var Ma=100,rM=101,aM=102,oM=103,lM=104,cM=200,uM=201,fM=202,hM=203,rv=204,av=205,dM=206,pM=207,mM=208,gM=209,vM=210,xM=211,yM=212,_M=213,SM=214,Kf=0,Go=1,Jf=2,Vo=3,jf=4,$f=5,eh=6,th=7,ov=0,AM=1,MM=2,gs=0,lv=1,cv=2,uv=3,fv=4,hv=5,dv=6,pv=7;var mv=300,Ea=301,Jo=302,Ch=303,Rh=304,nh=1e3,hs=1001,ih=1002,fn=1003,EM=1004;var Tc=1005;var Dt=1006,Dh=1007;var Xs=1008;var Gt=1009,gv=1010,vv=1011,jo=1012,Uh=1013,Wi=1014,fi=1015,Xi=1016,Ih=1017,Bh=1018,Ys=1020,xv=35902,yv=35899,_v=1021,Sv=1022,Yi=1023,ks=1026,qi=1027,Av=1028,Nh=1029,zr=1030,Lh=1031;var Ph=1033,bc=33776,wc=33777,Cc=33778,Rc=33779,Oh=35840,Fh=35841,zh=35842,Hh=35843,Gh=36196,Vh=37492,kh=37496,Wh=37488,Xh=37489,Dc=37490,Yh=37491,qh=37808,Qh=37809,Zh=37810,Kh=37811,Jh=37812,jh=37813,$h=37814,ed=37815,td=37816,nd=37817,id=37818,sd=37819,rd=37820,ad=37821,od=36492,ld=36494,cd=36495,ud=36283,fd=36284,Uc=36285,hd=36286;var uc=2300,sh=2301,Qf=2302,Zg=2303,Kg=2400,Jg=2401,jg=2402;var vs=3200;var Mv=0,TM=1,hi="",Tt="srgb",Ws="srgb-linear",fc="linear",ct="srgb";var Zf=7680;var bM=519,wM=512,CM=513,RM=514,Ev=515,DM=516,UM=517,Tv=518,bv=519,IM=35044;var Ic="300 es",Ci=2e3,hc=2001;function mw(t){for(let e=t.length-1;e>=0;--e)if(t[e]>=65535)return!0;return!1}function gw(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)}function dc(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function BM(){let t=dc("canvas");return t.style.display="block",t}var NA={},ko=null;function wv(...t){let e="THREE."+t.shift();ko?ko("log",e,...t):console.log(e,...t)}function NM(t){let e=t[0];if(typeof e=="string"&&e.startsWith("TSL:")){let n=t[1];n&&n.isStackTrace?t[0]+=" "+n.getLocation():t[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return t}function De(...t){t=NM(t);let e="THREE."+t.shift();if(ko)ko("warn",e,...t);else{let n=t[0];n&&n.isStackTrace?console.warn(n.getError(e)):console.warn(e,...t)}}function Be(...t){t=NM(t);let e="THREE."+t.shift();if(ko)ko("error",e,...t);else{let n=t[0];n&&n.isStackTrace?console.error(n.getError(e)):console.error(e,...t)}}function cc(...t){let e=t.join(" ");e in NA||(NA[e]=!0,De(...t))}function LM(t,e,n){return new Promise(function(i,s){function r(){switch(t.clientWaitSync(e,t.SYNC_FLUSH_COMMANDS_BIT,0)){case t.WAIT_FAILED:s();break;case t.TIMEOUT_EXPIRED:setTimeout(r,n);break;default:i()}}setTimeout(r,n)})}var PM={[Kf]:Kf,[Jf]:eh,[jf]:jf,[Vo]:$f,[Go]:Go,[eh]:Jf,[th]:th,[$f]:Vo},qn=class{addEventListener(e,n){this._listeners===void 0&&(this._listeners={});let i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(n)===-1&&i[e].push(n)}hasEventListener(e,n){let i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(n)!==-1}removeEventListener(e,n){let i=this._listeners;if(i===void 0)return;let s=i[e];if(s!==void 0){let r=s.indexOf(n);r!==-1&&s.splice(r,1)}}dispatchEvent(e){let n=this._listeners;if(n===void 0)return;let i=n[e.type];if(i!==void 0){e.target=this;let s=i.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,e);e.target=null}}},wn=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];var Cg=Math.PI/180,rh=180/Math.PI;function Bc(){let t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(wn[t&255]+wn[t>>8&255]+wn[t>>16&255]+wn[t>>24&255]+"-"+wn[e&255]+wn[e>>8&255]+"-"+wn[e>>16&15|64]+wn[e>>24&255]+"-"+wn[n&63|128]+wn[n>>8&255]+"-"+wn[n>>16&255]+wn[n>>24&255]+wn[i&255]+wn[i>>8&255]+wn[i>>16&255]+wn[i>>24&255]).toLowerCase()}function $e(t,e,n){return Math.max(e,Math.min(n,t))}function vw(t,e){return(t%e+e)%e}function Rg(t,e,n){return(1-n)*t+n*e}function OM(t){return Math.pow(2,Math.floor(Math.log(t)/Math.LN2))}function Cv(t){switch(t.constructor){case Uint32Array:return 4294967295;case Uint16Array:return 65535;case Uint8Array:case Uint8ClampedArray:return 255;case Int32Array:return 2147483647;case Int16Array:return 32767;case Int8Array:return 127;default:return}}function xw(t,e){if(e.constructor===Float32Array)return t;let n=Cv(e);if(n===void 0)throw new Error("THREE.MathUtils: Invalid component type.");return Math.max(t/n,-1)}function yw(t,e){if(e.constructor===Float32Array)return t;let n=Cv(e);if(n===void 0)throw new Error("THREE.MathUtils: Invalid component type.");return Math.round(t*n)}var Bv=class Bv{constructor(e=0,n=0){this.x=e,this.y=n}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,n){return this.x=e,this.y=n,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;default:throw new Error("THREE.Vector2: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){let n=this.x,i=this.y,s=e.elements;return this.x=s[0]*n+s[3]*i+s[6],this.y=s[1]*n+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,n){return this.x=$e(this.x,e.x,n.x),this.y=$e(this.y,e.y,n.y),this}clampScalar(e,n){return this.x=$e(this.x,e,n),this.y=$e(this.y,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar($e(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos($e(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y;return n*n+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this}rotateAround(e,n){let i=Math.cos(n),s=Math.sin(n),r=this.x-e.x,a=this.y-e.y;return this.x=r*i-a*s+e.x,this.y=r*s+a*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}};Bv.prototype.isVector2=!0;var Oe=Bv,ds=class{constructor(e=0,n=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=n,this._z=i,this._w=s}static slerpFlat(e,n,i,s,r,a,o){let l=i[s+0],c=i[s+1],h=i[s+2],p=i[s+3],u=r[a+0],d=r[a+1],x=r[a+2],A=r[a+3];if(p!==A||l!==u||c!==d||h!==x){let g=l*u+c*d+h*x+p*A;g<0&&(u=-u,d=-d,x=-x,A=-A,g=-g);let f=1-o;if(g<.9995){let m=Math.acos(g),y=Math.sin(m);f=Math.sin(f*m)/y,o=Math.sin(o*m)/y,l=l*f+u*o,c=c*f+d*o,h=h*f+x*o,p=p*f+A*o}else{l=l*f+u*o,c=c*f+d*o,h=h*f+x*o,p=p*f+A*o;let m=1/Math.sqrt(l*l+c*c+h*h+p*p);l*=m,c*=m,h*=m,p*=m}}e[n]=l,e[n+1]=c,e[n+2]=h,e[n+3]=p}static multiplyQuaternionsFlat(e,n,i,s,r,a){let o=i[s],l=i[s+1],c=i[s+2],h=i[s+3],p=r[a],u=r[a+1],d=r[a+2],x=r[a+3];return e[n]=o*x+h*p+l*d-c*u,e[n+1]=l*x+h*u+c*p-o*d,e[n+2]=c*x+h*d+o*u-l*p,e[n+3]=h*x-o*p-l*u-c*d,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,n,i,s){return this._x=e,this._y=n,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,n=!0){let i=e._x,s=e._y,r=e._z,a=e._order,o=Math.cos,l=Math.sin,c=o(i/2),h=o(s/2),p=o(r/2),u=l(i/2),d=l(s/2),x=l(r/2);switch(a){case"XYZ":this._x=u*h*p+c*d*x,this._y=c*d*p-u*h*x,this._z=c*h*x+u*d*p,this._w=c*h*p-u*d*x;break;case"YXZ":this._x=u*h*p+c*d*x,this._y=c*d*p-u*h*x,this._z=c*h*x-u*d*p,this._w=c*h*p+u*d*x;break;case"ZXY":this._x=u*h*p-c*d*x,this._y=c*d*p+u*h*x,this._z=c*h*x+u*d*p,this._w=c*h*p-u*d*x;break;case"ZYX":this._x=u*h*p-c*d*x,this._y=c*d*p+u*h*x,this._z=c*h*x-u*d*p,this._w=c*h*p+u*d*x;break;case"YZX":this._x=u*h*p+c*d*x,this._y=c*d*p+u*h*x,this._z=c*h*x-u*d*p,this._w=c*h*p-u*d*x;break;case"XZY":this._x=u*h*p-c*d*x,this._y=c*d*p-u*h*x,this._z=c*h*x+u*d*p,this._w=c*h*p+u*d*x;break;default:De("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return n===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,n){let i=n/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){let n=e.elements,i=n[0],s=n[4],r=n[8],a=n[1],o=n[5],l=n[9],c=n[2],h=n[6],p=n[10],u=i+o+p;if(u>0){let d=.5/Math.sqrt(u+1);this._w=.25/d,this._x=(h-l)*d,this._y=(r-c)*d,this._z=(a-s)*d}else if(i>o&&i>p){let d=2*Math.sqrt(1+i-o-p);this._w=(h-l)/d,this._x=.25*d,this._y=(s+a)/d,this._z=(r+c)/d}else if(o>p){let d=2*Math.sqrt(1+o-i-p);this._w=(r-c)/d,this._x=(s+a)/d,this._y=.25*d,this._z=(l+h)/d}else{let d=2*Math.sqrt(1+p-i-o);this._w=(a-s)/d,this._x=(r+c)/d,this._y=(l+h)/d,this._z=.25*d}return this._onChangeCallback(),this}setFromUnitVectors(e,n){let i=e.dot(n)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*n.z-e.z*n.y,this._y=e.z*n.x-e.x*n.z,this._z=e.x*n.y-e.y*n.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs($e(this.dot(e),-1,1)))}rotateTowards(e,n){let i=this.angleTo(e);if(i===0)return this;let s=Math.min(1,n/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=n._x,l=n._y,c=n._z,h=n._w;return this._x=i*h+a*o+s*c-r*l,this._y=s*h+a*l+r*o-i*c,this._z=r*h+a*c+i*l-s*o,this._w=a*h-i*o-s*l-r*c,this._onChangeCallback(),this}slerp(e,n){let i=e._x,s=e._y,r=e._z,a=e._w,o=this.dot(e);o<0&&(i=-i,s=-s,r=-r,a=-a,o=-o);let l=1-n;if(o<.9995){let c=Math.acos(o),h=Math.sin(c);l=Math.sin(l*c)/h,n=Math.sin(n*c)/h,this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this._onChangeCallback()}else this._x=this._x*l+i*n,this._y=this._y*l+s*n,this._z=this._z*l+r*n,this._w=this._w*l+a*n,this.normalize();return this}slerpQuaternions(e,n,i){return this.copy(e).slerp(n,i)}random(){let e=2*Math.PI*Math.random(),n=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(n),r*Math.cos(n))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,n=0){return this._x=e[n],this._y=e[n+1],this._z=e[n+2],this._w=e[n+3],this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._w,e}fromBufferAttribute(e,n){return this._x=e.getX(n),this._y=e.getY(n),this._z=e.getZ(n),this._w=e.getW(n),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},Nv=class Nv{constructor(e=0,n=0,i=0){this.x=e,this.y=n,this.z=i}set(e,n,i){return i===void 0&&(i=this.z),this.x=e,this.y=n,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;default:throw new Error("THREE.Vector3: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,n){return this.x=e.x*n.x,this.y=e.y*n.y,this.z=e.z*n.z,this}applyEuler(e){return this.applyQuaternion(LA.setFromEuler(e))}applyAxisAngle(e,n){return this.applyQuaternion(LA.setFromAxisAngle(e,n))}applyMatrix3(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[3]*i+r[6]*s,this.y=r[1]*n+r[4]*i+r[7]*s,this.z=r[2]*n+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=e.elements,a=1/(r[3]*n+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*n+r[4]*i+r[8]*s+r[12])*a,this.y=(r[1]*n+r[5]*i+r[9]*s+r[13])*a,this.z=(r[2]*n+r[6]*i+r[10]*s+r[14])*a,this}applyQuaternion(e){let n=this.x,i=this.y,s=this.z,r=e.x,a=e.y,o=e.z,l=e.w,c=2*(a*s-o*i),h=2*(o*n-r*s),p=2*(r*i-a*n);return this.x=n+l*c+a*p-o*h,this.y=i+l*h+o*c-r*p,this.z=s+l*p+r*h-a*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){let n=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*n+r[4]*i+r[8]*s,this.y=r[1]*n+r[5]*i+r[9]*s,this.z=r[2]*n+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,n){return this.x=$e(this.x,e.x,n.x),this.y=$e(this.y,e.y,n.y),this.z=$e(this.z,e.z,n.z),this}clampScalar(e,n){return this.x=$e(this.x,e,n),this.y=$e(this.y,e,n),this.z=$e(this.z,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar($e(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,n){let i=e.x,s=e.y,r=e.z,a=n.x,o=n.y,l=n.z;return this.x=s*l-r*o,this.y=r*a-i*l,this.z=i*o-s*a,this}projectOnVector(e){let n=e.lengthSq();if(n===0)return this.set(0,0,0);let i=e.dot(this)/n;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return Dg.copy(this).projectOnVector(e),this.sub(Dg)}reflect(e){return this.sub(Dg.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){let n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;let i=this.dot(e)/n;return Math.acos($e(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){let n=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return n*n+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,n,i){let s=Math.sin(n)*e;return this.x=s*Math.sin(i),this.y=Math.cos(n)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,n,i){return this.x=e*Math.sin(n),this.y=i,this.z=e*Math.cos(n),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this}setFromMatrixScale(e){let n=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=n,this.y=i,this.z=s,this}setFromMatrixColumn(e,n){return this.fromArray(e.elements,n*4)}setFromMatrix3Column(e,n){return this.fromArray(e.elements,n*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let e=Math.random()*Math.PI*2,n=Math.random()*2-1,i=Math.sqrt(1-n*n);return this.x=i*Math.cos(e),this.y=n,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}};Nv.prototype.isVector3=!0;var G=Nv,Dg=new G,LA=new ds,Lv=class Lv{constructor(e,n,i,s,r,a,o,l,c){this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c)}set(e,n,i,s,r,a,o,l,c){let h=this.elements;return h[0]=e,h[1]=s,h[2]=o,h[3]=n,h[4]=r,h[5]=l,h[6]=i,h[7]=a,h[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],this}extractBasis(e,n,i){return e.setFromMatrix3Column(this,0),n.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){let n=e.elements;return this.set(n[0],n[4],n[8],n[1],n[5],n[9],n[2],n[6],n[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[3],l=i[6],c=i[1],h=i[4],p=i[7],u=i[2],d=i[5],x=i[8],A=s[0],g=s[3],f=s[6],m=s[1],y=s[4],_=s[7],T=s[2],b=s[5],v=s[8];return r[0]=a*A+o*m+l*T,r[3]=a*g+o*y+l*b,r[6]=a*f+o*_+l*v,r[1]=c*A+h*m+p*T,r[4]=c*g+h*y+p*b,r[7]=c*f+h*_+p*v,r[2]=u*A+d*m+x*T,r[5]=u*g+d*y+x*b,r[8]=u*f+d*_+x*v,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[3]*=e,n[6]*=e,n[1]*=e,n[4]*=e,n[7]*=e,n[2]*=e,n[5]*=e,n[8]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8];return n*a*h-n*o*c-i*r*h+i*o*l+s*r*c-s*a*l}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=h*a-o*c,u=o*l-h*r,d=c*r-a*l,x=n*p+i*u+s*d;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);let A=1/x;return e[0]=p*A,e[1]=(s*c-h*i)*A,e[2]=(o*i-s*a)*A,e[3]=u*A,e[4]=(h*n-s*l)*A,e[5]=(s*r-o*n)*A,e[6]=d*A,e[7]=(i*l-c*n)*A,e[8]=(a*n-i*r)*A,this}transpose(){let e,n=this.elements;return e=n[1],n[1]=n[3],n[3]=e,e=n[2],n[2]=n[6],n[6]=e,e=n[5],n[5]=n[7],n[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){let n=this.elements;return e[0]=n[0],e[1]=n[3],e[2]=n[6],e[3]=n[1],e[4]=n[4],e[5]=n[7],e[6]=n[2],e[7]=n[5],e[8]=n[8],this}setUvTransform(e,n,i,s,r,a,o){let l=Math.cos(r),c=Math.sin(r);return this.set(i*l,i*c,-i*(l*a+c*o)+a+e,-s*c,s*l,-s*(-c*a+l*o)+o+n,0,0,1),this}scale(e,n){return cc("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(Ug.makeScale(e,n)),this}rotate(e){return cc("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(Ug.makeRotation(-e)),this}translate(e,n){return cc("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(Ug.makeTranslation(e,n)),this}makeTranslation(e,n){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,n,0,0,1),this}makeRotation(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,i,n,0,0,0,1),this}makeScale(e,n){return this.set(e,0,0,0,n,0,0,0,1),this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<9;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<9;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}};Lv.prototype.isMatrix3=!0;var Ne=Lv,Ug=new Ne,PA=new Ne().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),OA=new Ne().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function _w(){let t={enabled:!0,workingColorSpace:Ws,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===ct&&(s.r=Vs(s.r),s.g=Vs(s.g),s.b=Vs(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===ct&&(s.r=Ho(s.r),s.g=Ho(s.g),s.b=Ho(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===hi?fc:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[a].fromXYZ).multiply(this.spaces[r].toXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace}},e=[.64,.33,.3,.6,.15,.06],n=[.2126,.7152,.0722],i=[.3127,.329];return t.define({[Ws]:{primaries:e,whitePoint:i,transfer:fc,toXYZ:PA,fromXYZ:OA,luminanceCoefficients:n,workingColorSpaceConfig:{unpackColorSpace:Tt},outputColorSpaceConfig:{drawingBufferColorSpace:Tt}},[Tt]:{primaries:e,whitePoint:i,transfer:ct,toXYZ:PA,fromXYZ:OA,luminanceCoefficients:n,outputColorSpaceConfig:{drawingBufferColorSpace:Tt}}}),t}var je=_w();function Vs(t){return t<.04045?t*.0773993808:Math.pow(t*.9478672986+.0521327014,2.4)}function Ho(t){return t<.0031308?t*12.92:1.055*Math.pow(t,.41666)-.055}var wo,ah=class{static getDataURL(e,n="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{wo===void 0&&(wo=dc("canvas")),wo.width=e.width,wo.height=e.height;let s=wo.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=wo}return i.toDataURL(n)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){let n=dc("canvas");n.width=e.width,n.height=e.height;let i=n.getContext("2d");i.drawImage(e,0,0,e.width,e.height);let s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Vs(r[a]/255)*255;return i.putImageData(s,0,0),n}else if(e.data){let n=e.data.slice(0);for(let i=0;i<n.length;i++)n instanceof Uint8Array||n instanceof Uint8ClampedArray?n[i]=Math.floor(Vs(n[i]/255)*255):n[i]=Vs(n[i]);return{data:n,width:e.width,height:e.height}}else return De("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}},Sw=0,Wo=class{constructor(e=null){this.isTextureSource=!0,Object.defineProperty(this,"id",{value:Sw++}),this.uuid=Bc(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){let n=this.data;return typeof HTMLVideoElement<"u"&&n instanceof HTMLVideoElement?e.set(n.videoWidth,n.videoHeight,0):typeof VideoFrame<"u"&&n instanceof VideoFrame?e.set(n.displayWidth,n.displayHeight,0):n!==null?e.set(n.width||0,n.height||0,n.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.images[this.uuid]!==void 0)return e.images[this.uuid];let i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(Ig(s[a].image)):r.push(Ig(s[a]))}else r=Ig(s);i.url=r}return n||(e.images[this.uuid]=i),i}};function Ig(t){return typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap?ah.getDataURL(t):t.data?{data:Array.from(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(De("Texture: Unable to serialize Texture."),{})}var Aw=0,Bg=new G,qt=class t extends qn{constructor(e=t.DEFAULT_IMAGE,n=t.DEFAULT_MAPPING,i=hs,s=hs,r=Dt,a=Xs,o=Yi,l=Gt,c=t.DEFAULT_ANISOTROPY,h=hi){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Aw++}),this.uuid=Bc(),this.name="",this.source=new Wo(e),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=c,this.format=o,this.internalFormat=null,this.type=l,this.offset=new Oe(0,0),this.repeat=new Oe(1,1),this.center=new Oe(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Ne,this.generateMipmaps=!0,this.mipmapsAutoUpdate=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0,this.isPMREMTexture=!1,this.normalized=!1}get width(){return this.source.getSize(Bg).x}get height(){return this.source.getSize(Bg).y}get depth(){return this.source.getSize(Bg).z}get image(){return this.source.data}set image(e){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.normalized=e.normalized,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.mipmapsAutoUpdate=e.mipmapsAutoUpdate,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isPMREMTexture=e.isPMREMTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(let n in e){let i=e[n];if(i===void 0){De(`Texture.setValues(): parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){De(`Texture.setValues(): property '${n}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";if(!n&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];let i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,mipmapsAutoUpdate:this.mipmapsAutoUpdate,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),n||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==mv)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case nh:e.x=e.x-Math.floor(e.x);break;case hs:e.x=e.x<0?0:1;break;case ih:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case nh:e.y=e.y-Math.floor(e.y);break;case hs:e.y=e.y<0?0:1;break;case ih:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}};qt.DEFAULT_IMAGE=null;qt.DEFAULT_MAPPING=mv;qt.DEFAULT_ANISOTROPY=1;var Pv=class Pv{constructor(e=0,n=0,i=0,s=1){this.x=e,this.y=n,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,n,i,s){return this.x=e,this.y=n,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;case 3:this.w=n;break;default:throw new Error("THREE.Vector4: index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this.w=e.w+n.w,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this.w+=e.w*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this.w=e.w-n.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){let n=this.x,i=this.y,s=this.z,r=this.w,a=e.elements;return this.x=a[0]*n+a[4]*i+a[8]*s+a[12]*r,this.y=a[1]*n+a[5]*i+a[9]*s+a[13]*r,this.z=a[2]*n+a[6]*i+a[10]*s+a[14]*r,this.w=a[3]*n+a[7]*i+a[11]*s+a[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);let n=Math.sqrt(1-e.w*e.w);return n<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/n,this.y=e.y/n,this.z=e.z/n),this}setAxisAngleFromRotationMatrix(e){let n,i,s,r,l=e.elements,c=l[0],h=l[4],p=l[8],u=l[1],d=l[5],x=l[9],A=l[2],g=l[6],f=l[10];if(Math.abs(h-u)<.01&&Math.abs(p-A)<.01&&Math.abs(x-g)<.01){if(Math.abs(h+u)<.1&&Math.abs(p+A)<.1&&Math.abs(x+g)<.1&&Math.abs(c+d+f-3)<.1)return this.set(1,0,0,0),this;n=Math.PI;let y=(c+1)/2,_=(d+1)/2,T=(f+1)/2,b=(h+u)/4,v=(p+A)/4,E=(x+g)/4;return y>_&&y>T?y<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(y),s=b/i,r=v/i):_>T?_<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(_),i=b/s,r=E/s):T<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),i=v/r,s=E/r),this.set(i,s,r,n),this}let m=Math.sqrt((g-x)*(g-x)+(p-A)*(p-A)+(u-h)*(u-h));return Math.abs(m)<.001&&(m=1),this.x=(g-x)/m,this.y=(p-A)/m,this.z=(u-h)/m,this.w=Math.acos((c+d+f-1)/2),this}setFromMatrixPosition(e){let n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this.w=n[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,n){return this.x=$e(this.x,e.x,n.x),this.y=$e(this.y,e.y,n.y),this.z=$e(this.z,e.z,n.z),this.w=$e(this.w,e.w,n.w),this}clampScalar(e,n){return this.x=$e(this.x,e,n),this.y=$e(this.y,e,n),this.z=$e(this.z,e,n),this.w=$e(this.w,e,n),this}clampLength(e,n){let i=this.length();return this.divideScalar(i||1).multiplyScalar($e(i,e,n))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this.w+=(e.w-this.w)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this.w=e.w+(n.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this.w=e[n+3],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e[n+3]=this.w,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this.w=e.getW(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}};Pv.prototype.isVector4=!0;var Lt=Pv,oh=class extends qn{constructor(e=1,n=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Dt,depthBuffer:!0,stencilBuffer:!1,resolveColorBuffer:!0,resolveDepthBuffer:!0,resolveStencilBuffer:!0,storeMultisampledColorBuffer:!0,storeMultisampledDepthBuffer:!0,storeMultisampledStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},i),this.isRenderTarget=!0,this.width=e,this.height=n,this.depth=i.depth,this.scissor=new Lt(0,0,e,n),this.scissorTest=!1,this.viewport=new Lt(0,0,e,n),this.textures=[];let s=i.count;for(let r=0;r<s;r++)this.textures[r]=new qt({width:e,height:n,depth:i.depth}),this.textures[r].isRenderTargetTexture=!0,this.textures[r].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveColorBuffer=i.resolveColorBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this.storeMultisampledColorBuffer=i.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=i.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=i.storeMultisampledStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview,this.useArrayDepthTexture=i.useArrayDepthTexture}_setTextureOptions(e={}){let n={minFilter:Dt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(n.mapping=e.mapping),e.wrapS!==void 0&&(n.wrapS=e.wrapS),e.wrapT!==void 0&&(n.wrapT=e.wrapT),e.wrapR!==void 0&&(n.wrapR=e.wrapR),e.magFilter!==void 0&&(n.magFilter=e.magFilter),e.minFilter!==void 0&&(n.minFilter=e.minFilter),e.format!==void 0&&(n.format=e.format),e.type!==void 0&&(n.type=e.type),e.anisotropy!==void 0&&(n.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(n.colorSpace=e.colorSpace),e.flipY!==void 0&&(n.flipY=e.flipY),e.generateMipmaps!==void 0&&(n.generateMipmaps=e.generateMipmaps),e.mipmapsAutoUpdate!==void 0&&(n.mipmapsAutoUpdate=e.mipmapsAutoUpdate),e.internalFormat!==void 0&&(n.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(n)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&this._depthTexture.renderTarget===this&&(this._depthTexture.renderTarget=null),e!==null&&e.renderTarget===null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,n,i=1){if(this.width!==e||this.height!==n||this.depth!==i){this.width=e,this.height=n,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=n,this.textures[s].image.depth=i,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,n),this.scissor.set(0,0,e,n)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let n=0,i=e.textures.length;n<i;n++){this.textures[n]=e.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0,this.textures[n].renderTarget=this;let s=Object.assign({},e.textures[n].image);this.textures[n].source=new Wo(s)}if(this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveColorBuffer=e.resolveColorBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,this.storeMultisampledColorBuffer=e.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=e.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=e.storeMultisampledStencilBuffer,e.depthTexture!==null)if(e.depthTexture.renderTarget===e){let n=e.depthTexture.clone();n.renderTarget=null,this.depthTexture=n}else this.depthTexture=e.depthTexture;else this.depthTexture=null;return this.samples=e.samples,this.multiview=e.multiview,this.useArrayDepthTexture=e.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},zt=class extends oh{constructor(e=1,n=1,i={}){super(e,n,i),this.isWebGLRenderTarget=!0}},pc=class extends qt{constructor(e=null,n=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=fn,this.minFilter=fn,this.wrapR=hs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}copy(e){return super.copy(e),this.wrapR=e.wrapR,this}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}};var lh=class extends qt{constructor(e=null,n=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:n,height:i,depth:s},this.magFilter=fn,this.minFilter=fn,this.wrapR=hs,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}copy(e){return super.copy(e),this.wrapR=e.wrapR,this}};var wh=class wh{constructor(e,n,i,s,r,a,o,l,c,h,p,u,d,x,A,g){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,n,i,s,r,a,o,l,c,h,p,u,d,x,A,g)}set(e,n,i,s,r,a,o,l,c,h,p,u,d,x,A,g){let f=this.elements;return f[0]=e,f[4]=n,f[8]=i,f[12]=s,f[1]=r,f[5]=a,f[9]=o,f[13]=l,f[2]=c,f[6]=h,f[10]=p,f[14]=u,f[3]=d,f[7]=x,f[11]=A,f[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new wh().fromArray(this.elements)}copy(e){let n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],n[9]=i[9],n[10]=i[10],n[11]=i[11],n[12]=i[12],n[13]=i[13],n[14]=i[14],n[15]=i[15],this}copyPosition(e){let n=this.elements,i=e.elements;return n[12]=i[12],n[13]=i[13],n[14]=i[14],this}setFromMatrix3(e){let n=e.elements;return this.set(n[0],n[3],n[6],0,n[1],n[4],n[7],0,n[2],n[5],n[8],0,0,0,0,1),this}extractBasis(e,n,i){return this.determinantAffine()===0?(e.set(1,0,0),n.set(0,1,0),i.set(0,0,1),this):(e.setFromMatrixColumn(this,0),n.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this)}makeBasis(e,n,i){return this.set(e.x,n.x,i.x,0,e.y,n.y,i.y,0,e.z,n.z,i.z,0,0,0,0,1),this}extractRotation(e){if(e.determinantAffine()===0)return this.identity();let n=this.elements,i=e.elements,s=1/Co.setFromMatrixColumn(e,0).length(),r=1/Co.setFromMatrixColumn(e,1).length(),a=1/Co.setFromMatrixColumn(e,2).length();return n[0]=i[0]*s,n[1]=i[1]*s,n[2]=i[2]*s,n[3]=0,n[4]=i[4]*r,n[5]=i[5]*r,n[6]=i[6]*r,n[7]=0,n[8]=i[8]*a,n[9]=i[9]*a,n[10]=i[10]*a,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromEuler(e){let n=this.elements,i=e.x,s=e.y,r=e.z,a=Math.cos(i),o=Math.sin(i),l=Math.cos(s),c=Math.sin(s),h=Math.cos(r),p=Math.sin(r);if(e.order==="XYZ"){let u=a*h,d=a*p,x=o*h,A=o*p;n[0]=l*h,n[4]=-l*p,n[8]=c,n[1]=d+x*c,n[5]=u-A*c,n[9]=-o*l,n[2]=A-u*c,n[6]=x+d*c,n[10]=a*l}else if(e.order==="YXZ"){let u=l*h,d=l*p,x=c*h,A=c*p;n[0]=u+A*o,n[4]=x*o-d,n[8]=a*c,n[1]=a*p,n[5]=a*h,n[9]=-o,n[2]=d*o-x,n[6]=A+u*o,n[10]=a*l}else if(e.order==="ZXY"){let u=l*h,d=l*p,x=c*h,A=c*p;n[0]=u-A*o,n[4]=-a*p,n[8]=x+d*o,n[1]=d+x*o,n[5]=a*h,n[9]=A-u*o,n[2]=-a*c,n[6]=o,n[10]=a*l}else if(e.order==="ZYX"){let u=a*h,d=a*p,x=o*h,A=o*p;n[0]=l*h,n[4]=x*c-d,n[8]=u*c+A,n[1]=l*p,n[5]=A*c+u,n[9]=d*c-x,n[2]=-c,n[6]=o*l,n[10]=a*l}else if(e.order==="YZX"){let u=a*l,d=a*c,x=o*l,A=o*c;n[0]=l*h,n[4]=A-u*p,n[8]=x*p+d,n[1]=p,n[5]=a*h,n[9]=-o*h,n[2]=-c*h,n[6]=d*p+x,n[10]=u-A*p}else if(e.order==="XZY"){let u=a*l,d=a*c,x=o*l,A=o*c;n[0]=l*h,n[4]=-p,n[8]=c*h,n[1]=u*p+A,n[5]=a*h,n[9]=d*p-x,n[2]=x*p-d,n[6]=o*h,n[10]=A*p+u}return n[3]=0,n[7]=0,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromQuaternion(e){return this.compose(Mw,e,Ew)}lookAt(e,n,i){let s=this.elements;return ai.subVectors(e,n),ai.lengthSq()===0&&(ai.z=1),ai.normalize(),br.crossVectors(i,ai),br.lengthSq()===0&&(Math.abs(i.z)===1?ai.x+=1e-4:ai.z+=1e-4,ai.normalize(),br.crossVectors(i,ai)),br.normalize(),Cf.crossVectors(ai,br),s[0]=br.x,s[4]=Cf.x,s[8]=ai.x,s[1]=br.y,s[5]=Cf.y,s[9]=ai.y,s[2]=br.z,s[6]=Cf.z,s[10]=ai.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){let i=e.elements,s=n.elements,r=this.elements,a=i[0],o=i[4],l=i[8],c=i[12],h=i[1],p=i[5],u=i[9],d=i[13],x=i[2],A=i[6],g=i[10],f=i[14],m=i[3],y=i[7],_=i[11],T=i[15],b=s[0],v=s[4],E=s[8],N=s[12],C=s[1],B=s[5],F=s[9],X=s[13],D=s[2],V=s[6],q=s[10],H=s[14],te=s[3],k=s[7],$=s[11],ee=s[15];return r[0]=a*b+o*C+l*D+c*te,r[4]=a*v+o*B+l*V+c*k,r[8]=a*E+o*F+l*q+c*$,r[12]=a*N+o*X+l*H+c*ee,r[1]=h*b+p*C+u*D+d*te,r[5]=h*v+p*B+u*V+d*k,r[9]=h*E+p*F+u*q+d*$,r[13]=h*N+p*X+u*H+d*ee,r[2]=x*b+A*C+g*D+f*te,r[6]=x*v+A*B+g*V+f*k,r[10]=x*E+A*F+g*q+f*$,r[14]=x*N+A*X+g*H+f*ee,r[3]=m*b+y*C+_*D+T*te,r[7]=m*v+y*B+_*V+T*k,r[11]=m*E+y*F+_*q+T*$,r[15]=m*N+y*X+_*H+T*ee,this}multiplyScalar(e){let n=this.elements;return n[0]*=e,n[4]*=e,n[8]*=e,n[12]*=e,n[1]*=e,n[5]*=e,n[9]*=e,n[13]*=e,n[2]*=e,n[6]*=e,n[10]*=e,n[14]*=e,n[3]*=e,n[7]*=e,n[11]*=e,n[15]*=e,this}determinant(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[12],a=e[1],o=e[5],l=e[9],c=e[13],h=e[2],p=e[6],u=e[10],d=e[14],x=e[3],A=e[7],g=e[11],f=e[15],m=l*d-c*u,y=o*d-c*p,_=o*u-l*p,T=a*d-c*h,b=a*u-l*h,v=a*p-o*h;return n*(A*m-g*y+f*_)-i*(x*m-g*T+f*b)+s*(x*y-A*T+f*v)-r*(x*_-A*b+g*v)}determinantAffine(){let e=this.elements,n=e[0],i=e[4],s=e[8],r=e[1],a=e[5],o=e[9],l=e[2],c=e[6],h=e[10];return n*(a*h-o*c)-i*(r*h-o*l)+s*(r*c-a*l)}transpose(){let e=this.elements,n;return n=e[1],e[1]=e[4],e[4]=n,n=e[2],e[2]=e[8],e[8]=n,n=e[6],e[6]=e[9],e[9]=n,n=e[3],e[3]=e[12],e[12]=n,n=e[7],e[7]=e[13],e[13]=n,n=e[11],e[11]=e[14],e[14]=n,this}setPosition(e,n,i){let s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=n,s[14]=i),this}invert(){let e=this.elements,n=e[0],i=e[1],s=e[2],r=e[3],a=e[4],o=e[5],l=e[6],c=e[7],h=e[8],p=e[9],u=e[10],d=e[11],x=e[12],A=e[13],g=e[14],f=e[15],m=n*o-i*a,y=n*l-s*a,_=n*c-r*a,T=i*l-s*o,b=i*c-r*o,v=s*c-r*l,E=h*A-p*x,N=h*g-u*x,C=h*f-d*x,B=p*g-u*A,F=p*f-d*A,X=u*f-d*g,D=m*X-y*F+_*B+T*C-b*N+v*E;if(D===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let V=1/D;return e[0]=(o*X-l*F+c*B)*V,e[1]=(s*F-i*X-r*B)*V,e[2]=(A*v-g*b+f*T)*V,e[3]=(u*b-p*v-d*T)*V,e[4]=(l*C-a*X-c*N)*V,e[5]=(n*X-s*C+r*N)*V,e[6]=(g*_-x*v-f*y)*V,e[7]=(h*v-u*_+d*y)*V,e[8]=(a*F-o*C+c*E)*V,e[9]=(i*C-n*F-r*E)*V,e[10]=(x*b-A*_+f*m)*V,e[11]=(p*_-h*b-d*m)*V,e[12]=(o*N-a*B-l*E)*V,e[13]=(n*B-i*N+s*E)*V,e[14]=(A*y-x*T-g*m)*V,e[15]=(h*T-p*y+u*m)*V,this}scale(e){let n=this.elements,i=e.x,s=e.y,r=e.z;return n[0]*=i,n[4]*=s,n[8]*=r,n[1]*=i,n[5]*=s,n[9]*=r,n[2]*=i,n[6]*=s,n[10]*=r,n[3]*=i,n[7]*=s,n[11]*=r,this}getMaxScaleOnAxis(){let e=this.elements,n=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(n,i,s))}makeTranslation(e,n,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,n,0,0,1,i,0,0,0,1),this}makeRotationX(e){let n=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,n,-i,0,0,i,n,0,0,0,0,1),this}makeRotationY(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,0,i,0,0,1,0,0,-i,0,n,0,0,0,0,1),this}makeRotationZ(e){let n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,0,i,n,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,n){let i=Math.cos(n),s=Math.sin(n),r=1-i,a=e.x,o=e.y,l=e.z,c=r*a,h=r*o;return this.set(c*a+i,c*o-s*l,c*l+s*o,0,c*o+s*l,h*o+i,h*l-s*a,0,c*l-s*o,h*l+s*a,r*l*l+i,0,0,0,0,1),this}makeScale(e,n,i){return this.set(e,0,0,0,0,n,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,n,i,s,r,a){return this.set(1,i,r,0,e,1,a,0,n,s,1,0,0,0,0,1),this}compose(e,n,i){let s=this.elements,r=n._x,a=n._y,o=n._z,l=n._w,c=r+r,h=a+a,p=o+o,u=r*c,d=r*h,x=r*p,A=a*h,g=a*p,f=o*p,m=l*c,y=l*h,_=l*p,T=i.x,b=i.y,v=i.z;return s[0]=(1-(A+f))*T,s[1]=(d+_)*T,s[2]=(x-y)*T,s[3]=0,s[4]=(d-_)*b,s[5]=(1-(u+f))*b,s[6]=(g+m)*b,s[7]=0,s[8]=(x+y)*v,s[9]=(g-m)*v,s[10]=(1-(u+A))*v,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,n,i){let s=this.elements;e.x=s[12],e.y=s[13],e.z=s[14];let r=this.determinantAffine();if(r===0)return i.set(1,1,1),n.identity(),this;let a=Co.set(s[0],s[1],s[2]).length(),o=Co.set(s[4],s[5],s[6]).length(),l=Co.set(s[8],s[9],s[10]).length();r<0&&(a=-a),Fi.copy(this);let c=1/a,h=1/o,p=1/l;return Fi.elements[0]*=c,Fi.elements[1]*=c,Fi.elements[2]*=c,Fi.elements[4]*=h,Fi.elements[5]*=h,Fi.elements[6]*=h,Fi.elements[8]*=p,Fi.elements[9]*=p,Fi.elements[10]*=p,n.setFromRotationMatrix(Fi),i.x=a,i.y=o,i.z=l,this}makePerspective(e,n,i,s,r,a,o=Ci,l=!1){let c=this.elements,h=2*r/(n-e),p=2*r/(i-s),u=(n+e)/(n-e),d=(i+s)/(i-s),x,A;if(l)x=r/(a-r),A=a*r/(a-r);else if(o===Ci)x=-(a+r)/(a-r),A=-2*a*r/(a-r);else if(o===hc)x=-a/(a-r),A=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=u,c[12]=0,c[1]=0,c[5]=p,c[9]=d,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=A,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,n,i,s,r,a,o=Ci,l=!1){let c=this.elements,h=2/(n-e),p=2/(i-s),u=-(n+e)/(n-e),d=-(i+s)/(i-s),x,A;if(l)x=1/(a-r),A=a/(a-r);else if(o===Ci)x=-2/(a-r),A=-(a+r)/(a-r);else if(o===hc)x=-1/(a-r),A=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return c[0]=h,c[4]=0,c[8]=0,c[12]=u,c[1]=0,c[5]=p,c[9]=0,c[13]=d,c[2]=0,c[6]=0,c[10]=x,c[14]=A,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){let n=this.elements,i=e.elements;for(let s=0;s<16;s++)if(n[s]!==i[s])return!1;return!0}fromArray(e,n=0){for(let i=0;i<16;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){let i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e[n+9]=i[9],e[n+10]=i[10],e[n+11]=i[11],e[n+12]=i[12],e[n+13]=i[13],e[n+14]=i[14],e[n+15]=i[15],e}};wh.prototype.isMatrix4=!0;var Ft=wh,Co=new G,Fi=new Ft,Mw=new G(0,0,0),Ew=new G(1,1,1),br=new G,Cf=new G,ai=new G,FA=new Ft,zA=new ds,Ir=class t{constructor(e=0,n=0,i=0,s=t.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=n,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,n,i,s=this._order){return this._x=e,this._y=n,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,n=this._order,i=!0){let s=e.elements,r=s[0],a=s[4],o=s[8],l=s[1],c=s[5],h=s[9],p=s[2],u=s[6],d=s[10];switch(n){case"XYZ":this._y=Math.asin($e(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,d),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(u,c),this._z=0);break;case"YXZ":this._x=Math.asin(-$e(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,d),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-p,r),this._z=0);break;case"ZXY":this._x=Math.asin($e(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-p,d),this._z=Math.atan2(-a,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-$e(p,-1,1)),Math.abs(p)<.9999999?(this._x=Math.atan2(u,d),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-a,c));break;case"YZX":this._z=Math.asin($e(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-h,c),this._y=Math.atan2(-p,r)):(this._x=0,this._y=Math.atan2(o,d));break;case"XZY":this._z=Math.asin(-$e(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(u,c),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,d),this._y=0);break;default:De("Euler: .setFromRotationMatrix() encountered an unknown order: "+n)}return this._order=n,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,n,i){return FA.makeRotationFromQuaternion(e),this.setFromRotationMatrix(FA,n,i)}setFromVector3(e,n=this._order){return this.set(e.x,e.y,e.z,n)}reorder(e){return zA.setFromEuler(this),this.setFromQuaternion(zA,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};Ir.DEFAULT_ORDER="XYZ";var mc=class{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}},Tw=0,HA=new G,Ro=new ds,Os=new Ft,Rf=new G,rc=new G,bw=new G,ww=new ds,GA=new G(1,0,0),VA=new G(0,1,0),kA=new G(0,0,1),WA={type:"added"},Cw={type:"removed"},Do={type:"childadded",child:null},Ng={type:"childremoved",child:null},Di=class t extends qn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Tw++}),this.uuid=Bc(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=t.DEFAULT_UP.clone();let e=new G,n=new Ir,i=new ds,s=new G(1,1,1);function r(){i.setFromEuler(n,!1)}function a(){n.setFromQuaternion(i,void 0,!1)}n._onChange(r),i._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new Ft},normalMatrix:{value:new Ne}}),this.matrix=new Ft,this.matrixWorld=new Ft,this.matrixAutoUpdate=t.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=t.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new mc,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale),this.pivot!==null&&this.position.copy(this.pivot).applyMatrix4(this.matrix).sub(this.pivot)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,n){this.quaternion.setFromAxisAngle(e,n)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,n){return Ro.setFromAxisAngle(e,n),this.quaternion.multiply(Ro),this}rotateOnWorldAxis(e,n){return Ro.setFromAxisAngle(e,n),this.quaternion.premultiply(Ro),this}rotateX(e){return this.rotateOnAxis(GA,e)}rotateY(e){return this.rotateOnAxis(VA,e)}rotateZ(e){return this.rotateOnAxis(kA,e)}translateOnAxis(e,n){return HA.copy(e).applyQuaternion(this.quaternion),this.position.add(HA.multiplyScalar(n)),this}translateX(e){return this.translateOnAxis(GA,e)}translateY(e){return this.translateOnAxis(VA,e)}translateZ(e){return this.translateOnAxis(kA,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Os.copy(this.matrixWorld).invert())}lookAt(e,n,i){e.isVector3?Rf.copy(e):Rf.set(e,n,i);let s=this.parent;this.updateWorldMatrix(!0,!1),rc.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Os.lookAt(rc,Rf,this.up):Os.lookAt(Rf,rc,this.up),this.quaternion.setFromRotationMatrix(Os),s&&(Os.extractRotation(s.matrixWorld),Ro.setFromRotationMatrix(Os),this.quaternion.premultiply(Ro.invert()))}add(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.add(arguments[n]);return this}return e===this?(Be("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(WA),Do.child=e,this.dispatchEvent(Do),Do.child=null):Be("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}let n=this.children.indexOf(e);return n!==-1&&(e.parent=null,this.children.splice(n,1),e.dispatchEvent(Cw),Ng.child=e,this.dispatchEvent(Ng),Ng.child=null),this}removeFromParent(){let e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Os.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Os.multiply(e.parent.matrixWorld)),e.applyMatrix4(Os),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(WA),Do.child=e,this.dispatchEvent(Do),Do.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,n){if(this[e]===n)return this;for(let i=0,s=this.children.length;i<s;i++){let a=this.children[i].getObjectByProperty(e,n);if(a!==void 0)return a}}getObjectsByProperty(e,n,i=[]){this[e]===n&&i.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(e,n,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(rc,e,bw),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(rc,ww,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);let n=this.matrixWorld.elements;return e.set(n[8],n[9],n[10]).normalize()}raycast(){}intersectsFrustum(){return!0}traverse(e){e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].traverseVisible(e)}traverseAncestors(e){let n=this.parent;n!==null&&(e(n),n.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let e=this.pivot;if(e!==null){let n=e.x,i=e.y,s=e.z,r=this.matrix.elements;r[12]+=n-r[0]*n-r[4]*i-r[8]*s,r[13]+=i-r[1]*n-r[5]*i-r[9]*s,r[14]+=s-r[2]*n-r[6]*i-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);let n=this.children;for(let i=0,s=n.length;i<s;i++)n[i].updateMatrixWorld(e)}updateWorldMatrix(e,n,i=!1){let s=this.parent;if(e===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||i)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,i=!0),n===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,i)}}toJSON(e){let n=e===void 0||typeof e=="string",i={};n&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,s.name=this.name,s.castShadow=this.castShadow,s.receiveShadow=this.receiveShadow,s.visible=this.visible,s.frustumCulled=this.frustumCulled,s.renderOrder=this.renderOrder,s.static=this.static,s.matrixAutoUpdate=this.matrixAutoUpdate,Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,l){return o[l.uuid]===void 0&&(o[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let l=o.shapes;if(Array.isArray(l))for(let c=0,h=l.length;c<h;c++){let p=l[c];r(e.shapes,p)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let l=0,c=this.material.length;l<c;l++)o.push(r(e.materials,this.material[l]));s.material=o}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let l=this.animations[o];s.animations.push(r(e.animations,l))}}if(n){let o=a(e.geometries),l=a(e.materials),c=a(e.textures),h=a(e.images),p=a(e.shapes),u=a(e.skeletons),d=a(e.animations),x=a(e.nodes);o.length>0&&(i.geometries=o),l.length>0&&(i.materials=l),c.length>0&&(i.textures=c),h.length>0&&(i.images=h),p.length>0&&(i.shapes=p),u.length>0&&(i.skeletons=u),d.length>0&&(i.animations=d),x.length>0&&(i.nodes=x)}return i.object=s,i;function a(o){let l=[];for(let c in o){let h=o[c];delete h.metadata,l.push(h)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,n=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.pivot=e.pivot!==null?e.pivot.clone():null,this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),n===!0)for(let i=0;i<e.children.length;i++){let s=e.children[i];this.add(s.clone())}return this}dispose(){this.dispatchEvent({type:"dispose"})}};Di.DEFAULT_UP=new G(0,1,0);Di.DEFAULT_MATRIX_AUTO_UPDATE=!0;Di.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var _a=class extends Di{constructor(){super(),this.isGroup=!0,this.type="Group"}},Rw={type:"move"},Xo=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new _a,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new _a,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new G,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new G),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new _a,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new G,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new G,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){let n=this._hand;if(n)for(let i of e.hand.values())this._getHandJoint(n,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,n,i){let s=null,r=null,a=null,o=this._targetRay,l=this._grip,c=this._hand;if(e&&n.session.visibilityState!=="visible-blurred"){if(c&&e.hand){a=!0;for(let A of e.hand.values()){let g=n.getJointPose(A,i),f=this._getHandJoint(c,A);g!==null&&(f.matrix.fromArray(g.transform.matrix),f.matrix.decompose(f.position,f.quaternion,f.scale),f.matrixWorldNeedsUpdate=!0,f.jointRadius=g.radius),f.visible=g!==null}let h=c.joints["index-finger-tip"],p=c.joints["thumb-tip"],u=h.position.distanceTo(p.position),d=.02,x=.005;c.inputState.pinching&&u>d+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&u<=d-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=n.getPose(e.gripSpace,i),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.quaternion,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1,l.eventsEnabled&&l.dispatchEvent({type:"gripUpdated",data:e,target:this})));o!==null&&(s=n.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.quaternion,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(Rw)))}return o!==null&&(o.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=a!==null),this}_getHandJoint(e,n){if(e.joints[n.jointName]===void 0){let i=new _a;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[n.jointName]=i,e.add(i)}return e.joints[n.jointName]}},FM={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},wr={h:0,s:0,l:0},Df={h:0,s:0,l:0};function Lg(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<1/6?t+(e-t)*6*n:n<1/2?e:n<2/3?t+(e-t)*6*(2/3-n):t}var Ye=class{constructor(e,n,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,n,i)}set(e,n,i){if(n===void 0&&i===void 0){let s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,n,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,n=Tt){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,je.colorSpaceToWorking(this,n),this}setRGB(e,n,i,s=je.workingColorSpace){return this.r=e,this.g=n,this.b=i,je.colorSpaceToWorking(this,s),this}setHSL(e,n,i,s=je.workingColorSpace){if(e=vw(e,1),n=$e(n,0,1),i=$e(i,0,1),n===0)this.r=this.g=this.b=i;else{let r=i<=.5?i*(1+n):i+n-i*n,a=2*i-r;this.r=Lg(a,r,e+1/3),this.g=Lg(a,r,e),this.b=Lg(a,r,e-1/3)}return je.colorSpaceToWorking(this,s),this}setStyle(e,n=Tt){function i(r){r!==void 0&&parseFloat(r)<1&&De("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,n);if(r=/^\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return i(r[4]),this.setRGB(Math.min(100,parseFloat(r[1]))/100,Math.min(100,parseFloat(r[2]))/100,Math.min(100,parseFloat(r[3]))/100,n);break;case"hsl":case"hsla":if(r=/^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,n);break;default:De("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,n);if(a===6)return this.setHex(parseInt(r,16),n);De("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,n);return this}setColorName(e,n=Tt){let i=FM[e.toLowerCase()];return i!==void 0?this.setHex(i,n):De("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Vs(e.r),this.g=Vs(e.g),this.b=Vs(e.b),this}copyLinearToSRGB(e){return this.r=Ho(e.r),this.g=Ho(e.g),this.b=Ho(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Tt){return je.workingToColorSpace(Cn.copy(this),e),Math.round($e(Cn.r*255,0,255))*65536+Math.round($e(Cn.g*255,0,255))*256+Math.round($e(Cn.b*255,0,255))}getHexString(e=Tt){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,n=je.workingColorSpace){je.workingToColorSpace(Cn.copy(this),n);let i=Cn.r,s=Cn.g,r=Cn.b,a=Math.max(i,s,r),o=Math.min(i,s,r),l,c,h=(o+a)/2;if(o===a)l=0,c=0;else{let p=a-o;switch(c=h<=.5?p/(a+o):p/(2-a-o),a){case i:l=(s-r)/p+(s<r?6:0);break;case s:l=(r-i)/p+2;break;case r:l=(i-s)/p+4;break}l/=6}return e.h=l,e.s=c,e.l=h,e}getRGB(e,n=je.workingColorSpace){return je.workingToColorSpace(Cn.copy(this),n),e.r=Cn.r,e.g=Cn.g,e.b=Cn.b,e}getStyle(e=Tt){je.workingToColorSpace(Cn.copy(this),e);let n=Cn.r,i=Cn.g,s=Cn.b;return e!==Tt?`color(${e} ${n.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(n*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,n,i){return this.getHSL(wr),this.setHSL(wr.h+e,wr.s+n,wr.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,n){return this.r=e.r+n.r,this.g=e.g+n.g,this.b=e.b+n.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,n){return this.r+=(e.r-this.r)*n,this.g+=(e.g-this.g)*n,this.b+=(e.b-this.b)*n,this}lerpColors(e,n,i){return this.r=e.r+(n.r-e.r)*i,this.g=e.g+(n.g-e.g)*i,this.b=e.b+(n.b-e.b)*i,this}lerpHSL(e,n){this.getHSL(wr),e.getHSL(Df);let i=Rg(wr.h,Df.h,n),s=Rg(wr.s,Df.s,n),r=Rg(wr.l,Df.l,n);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){let n=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*n+r[3]*i+r[6]*s,this.g=r[1]*n+r[4]*i+r[7]*s,this.b=r[2]*n+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,n=0){return this.r=e[n],this.g=e[n+1],this.b=e[n+2],this}toArray(e=[],n=0){return e[n]=this.r,e[n+1]=this.g,e[n+2]=this.b,e}fromBufferAttribute(e,n){return this.r=e.getX(n),this.g=e.getY(n),this.b=e.getZ(n),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},Cn=new Ye;Ye.NAMES=FM;var Br=class extends Di{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Ir,this.environmentIntensity=1,this.environmentRotation=new Ir,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,n){return super.copy(e,n),this.background=e.background!==null?e.background.clone():null,this.environment=e.environment!==null?e.environment.clone():null,this.fog=e.fog!==null?e.fog.clone():null,this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),this.overrideMaterial=e.overrideMaterial!==null?e.overrideMaterial.clone():null,this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){let n=super.toJSON(e);return this.fog!==null&&(n.object.fog=this.fog.toJSON()),n.object.backgroundBlurriness=this.backgroundBlurriness,n.object.backgroundIntensity=this.backgroundIntensity,n.object.backgroundRotation=this.backgroundRotation.toArray(),n.object.environmentIntensity=this.environmentIntensity,n.object.environmentRotation=this.environmentRotation.toArray(),n}},zi=new G,Fs=new G,Pg=new G,zs=new G,Uo=new G,Io=new G,XA=new G,Og=new G,Fg=new G,zg=new G,Hg=new Lt,Gg=new Lt,Vg=new Lt,Ur=class t{constructor(e=new G,n=new G,i=new G){this.a=e,this.b=n,this.c=i}static getNormal(e,n,i,s){s.subVectors(i,n),zi.subVectors(e,n),s.cross(zi);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,n,i,s,r){zi.subVectors(s,n),Fs.subVectors(i,n),Pg.subVectors(e,n);let a=zi.dot(zi),o=zi.dot(Fs),l=zi.dot(Pg),c=Fs.dot(Fs),h=Fs.dot(Pg),p=a*c-o*o;if(p===0)return r.set(0,0,0),null;let u=1/p,d=(c*l-o*h)*u,x=(a*h-o*l)*u;return r.set(1-d-x,x,d)}static containsPoint(e,n,i,s){return this.getBarycoord(e,n,i,s,zs)===null?!1:zs.x>=0&&zs.y>=0&&zs.x+zs.y<=1}static getInterpolation(e,n,i,s,r,a,o,l){return this.getBarycoord(e,n,i,s,zs)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,zs.x),l.addScaledVector(a,zs.y),l.addScaledVector(o,zs.z),l)}static getInterpolatedAttribute(e,n,i,s,r,a){return Hg.setScalar(0),Gg.setScalar(0),Vg.setScalar(0),Hg.fromBufferAttribute(e,n),Gg.fromBufferAttribute(e,i),Vg.fromBufferAttribute(e,s),a.setScalar(0),a.addScaledVector(Hg,r.x),a.addScaledVector(Gg,r.y),a.addScaledVector(Vg,r.z),a}static isFrontFacing(e,n,i,s){return zi.subVectors(i,n),Fs.subVectors(e,n),zi.cross(Fs).dot(s)<0}set(e,n,i){return this.a.copy(e),this.b.copy(n),this.c.copy(i),this}setFromPointsAndIndices(e,n,i,s){return this.a.copy(e[n]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,n,i,s){return this.a.fromBufferAttribute(e,n),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return zi.subVectors(this.c,this.b),Fs.subVectors(this.a,this.b),zi.cross(Fs).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return t.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,n){return t.getBarycoord(e,this.a,this.b,this.c,n)}getInterpolation(e,n,i,s,r){return t.getInterpolation(e,this.a,this.b,this.c,n,i,s,r)}containsPoint(e){return t.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return t.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,n){let i=this.a,s=this.b,r=this.c,a,o;Uo.subVectors(s,i),Io.subVectors(r,i),Og.subVectors(e,i);let l=Uo.dot(Og),c=Io.dot(Og);if(l<=0&&c<=0)return n.copy(i);Fg.subVectors(e,s);let h=Uo.dot(Fg),p=Io.dot(Fg);if(h>=0&&p<=h)return n.copy(s);let u=l*p-h*c;if(u<=0&&l>=0&&h<=0&&l-h>0)return a=l/(l-h),n.copy(i).addScaledVector(Uo,a);zg.subVectors(e,r);let d=Uo.dot(zg),x=Io.dot(zg);if(x>=0&&d<=x)return n.copy(r);let A=d*c-l*x;if(A<=0&&c>=0&&x<=0)return o=c/(c-x),n.copy(i).addScaledVector(Io,o);let g=h*x-d*p;if(g<=0&&p-h>=0&&d-x>=0)return XA.subVectors(r,s),o=(p-h)/(p-h+(d-x)),n.copy(s).addScaledVector(XA,o);let f=1/(g+A+u);return a=A*f,o=u*f,n.copy(i).addScaledVector(Uo,a).addScaledVector(Io,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},Nr=class{constructor(e=new G(1/0,1/0,1/0),n=new G(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=n}set(e,n){return this.min.copy(e),this.max.copy(n),this}setFromArray(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n+=3)this.expandByPoint(Hi.fromArray(e,n));return this}setFromBufferAttribute(e){this.makeEmpty();for(let n=0,i=e.count;n<i;n++)this.expandByPoint(Hi.fromBufferAttribute(e,n));return this}setFromPoints(e){this.makeEmpty();for(let n=0,i=e.length;n<i;n++)this.expandByPoint(e[n]);return this}setFromCenterAndSize(e,n){let i=Hi.copy(n).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,n=!1){return this.makeEmpty(),this.expandByObject(e,n)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,n=!1){e.updateWorldMatrix(!1,!1);let i=e.geometry;if(i!==void 0){let r=i.getAttribute("position");if(n===!0&&r!==void 0&&e.isInstancedMesh!==!0&&e.isBatchedMesh!==!0)for(let a=0,o=r.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,Hi):Hi.fromBufferAttribute(r,a),Hi.applyMatrix4(e.matrixWorld),this.expandByPoint(Hi);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Uf.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),Uf.copy(i.boundingBox)),Uf.applyMatrix4(e.matrixWorld),this.union(Uf)}let s=e.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],n);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,n){return n.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Hi),Hi.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let n,i;return e.normal.x>0?(n=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(n=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(n+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(n+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(n+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(n+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),n<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(ac),If.subVectors(this.max,ac),Bo.subVectors(e.a,ac),No.subVectors(e.b,ac),Lo.subVectors(e.c,ac),Cr.subVectors(No,Bo),Rr.subVectors(Lo,No),ga.subVectors(Bo,Lo);let n=[0,-Cr.z,Cr.y,0,-Rr.z,Rr.y,0,-ga.z,ga.y,Cr.z,0,-Cr.x,Rr.z,0,-Rr.x,ga.z,0,-ga.x,-Cr.y,Cr.x,0,-Rr.y,Rr.x,0,-ga.y,ga.x,0];return!kg(n,Bo,No,Lo,If)||(n=[1,0,0,0,1,0,0,0,1],!kg(n,Bo,No,Lo,If))?!1:(Bf.crossVectors(Cr,Rr),n=[Bf.x,Bf.y,Bf.z],kg(n,Bo,No,Lo,If))}clampPoint(e,n){return n.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Hi).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Hi).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Hs[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Hs[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Hs[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Hs[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Hs[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Hs[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Hs[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Hs[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Hs),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}},Hs=[new G,new G,new G,new G,new G,new G,new G,new G],Hi=new G,Uf=new Nr,Bo=new G,No=new G,Lo=new G,Cr=new G,Rr=new G,ga=new G,ac=new G,If=new G,Bf=new G,va=new G;function kg(t,e,n,i,s){for(let r=0,a=t.length-3;r<=a;r+=3){va.fromArray(t,r);let o=s.x*Math.abs(va.x)+s.y*Math.abs(va.y)+s.z*Math.abs(va.z),l=e.dot(va),c=n.dot(va),h=i.dot(va);if(Math.max(-Math.max(l,c,h),Math.min(l,c,h))>o)return!1}return!0}var Yt=new G,Nf=new Oe,Dw=0,li=class extends qn{constructor(e,n,i=!1){if(super(),Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Dw++}),this.name="",this.array=e,this.itemSize=n,this.count=e!==void 0?e.length/n:0,this.normalized=i,this.usage=IM,this.updateRanges=[],this.gpuType=fi,this.version=0,this._normalizationScale=e!==void 0?Cv(e):void 0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,n){this.updateRanges.push({start:e,count:n})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this._normalizationScale=e._normalizationScale,this}copyAt(e,n,i){e*=this.itemSize,i*=n.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=n.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let n=0,i=this.count;n<i;n++)Nf.fromBufferAttribute(this,n),Nf.applyMatrix3(e),this.setXY(n,Nf.x,Nf.y);else if(this.itemSize===3)for(let n=0,i=this.count;n<i;n++)Yt.fromBufferAttribute(this,n),Yt.applyMatrix3(e),this.setXYZ(n,Yt.x,Yt.y,Yt.z);return this}applyMatrix4(e){for(let n=0,i=this.count;n<i;n++)Yt.fromBufferAttribute(this,n),Yt.applyMatrix4(e),this.setXYZ(n,Yt.x,Yt.y,Yt.z);return this}applyNormalMatrix(e){for(let n=0,i=this.count;n<i;n++)Yt.fromBufferAttribute(this,n),Yt.applyNormalMatrix(e),this.setXYZ(n,Yt.x,Yt.y,Yt.z);return this}transformDirection(e){for(let n=0,i=this.count;n<i;n++)Yt.fromBufferAttribute(this,n),Yt.transformDirection(e),this.setXYZ(n,Yt.x,Yt.y,Yt.z);return this}set(e,n=0){return this.array.set(e,n),this}getComponent(e,n){let i=this.array[e*this.itemSize+n];return this.normalized&&(i=this._denormalize(i)),i}setComponent(e,n,i){return this.normalized&&(i=this._normalize(i)),this.array[e*this.itemSize+n]=i,this}getX(e){let n=this.array[e*this.itemSize];return this.normalized&&(n=this._denormalize(n)),n}setX(e,n){return this.normalized&&(n=this._normalize(n)),this.array[e*this.itemSize]=n,this}getY(e){let n=this.array[e*this.itemSize+1];return this.normalized&&(n=this._denormalize(n)),n}setY(e,n){return this.normalized&&(n=this._normalize(n)),this.array[e*this.itemSize+1]=n,this}getZ(e){let n=this.array[e*this.itemSize+2];return this.normalized&&(n=this._denormalize(n)),n}setZ(e,n){return this.normalized&&(n=this._normalize(n)),this.array[e*this.itemSize+2]=n,this}getW(e){let n=this.array[e*this.itemSize+3];return this.normalized&&(n=this._denormalize(n)),n}setW(e,n){return this.normalized&&(n=this._normalize(n)),this.array[e*this.itemSize+3]=n,this}setXY(e,n,i){return e*=this.itemSize,this.normalized&&(n=this._normalize(n),i=this._normalize(i)),this.array[e+0]=n,this.array[e+1]=i,this}setXYZ(e,n,i,s){return e*=this.itemSize,this.normalized&&(n=this._normalize(n),i=this._normalize(i),s=this._normalize(s)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,n,i,s,r){return e*=this.itemSize,this.normalized&&(n=this._normalize(n),i=this._normalize(i),s=this._normalize(s),r=this._normalize(r)),this.array[e+0]=n,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return e.name=this.name,e.usage=this.usage,e.gpuType=this.gpuType,e}dispose(){this.dispatchEvent({type:"dispose"})}_denormalize(e){let n=this._normalizationScale;return n!==void 0?Math.max(e/n,-1):xw(e,this.array)}_normalize(e){let n=this._normalizationScale;return n!==void 0?Math.round(e*n):yw(e,this.array)}};var gc=class extends li{constructor(e,n,i){super(new Uint16Array(e),n,i)}};var vc=class extends li{constructor(e,n,i){super(new Uint32Array(e),n,i)}};var Ri=class extends li{constructor(e,n,i){super(new Float32Array(e),n,i)}},Uw=new Nr,oc=new G,Wg=new G,Yo=class{constructor(e=new G,n=-1){this.isSphere=!0,this.center=e,this.radius=n}set(e,n){return this.center.copy(e),this.radius=n,this}setFromPoints(e,n){let i=this.center;n!==void 0?i.copy(n):Uw.setFromPoints(e).getCenter(i);let s=0;for(let r=0,a=e.length;r<a;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){let n=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=n*n}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,n){let i=this.center.distanceToSquared(e);return n.copy(e),i>this.radius*this.radius&&(n.sub(this.center).normalize(),n.multiplyScalar(this.radius).add(this.center)),n}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;oc.subVectors(e,this.center);let n=oc.lengthSq();if(n>this.radius*this.radius){let i=Math.sqrt(n),s=(i-this.radius)*.5;this.center.addScaledVector(oc,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Wg.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(oc.copy(e.center).add(Wg)),this.expandByPoint(oc.copy(e.center).sub(Wg))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}},Iw=0,wi=new Ft,Xg=new Di,Po=new G,oi=new Nr,lc=new Nr,un=new G,ps=class t extends qn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Iw++}),this.uuid=Bc(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(mw(e)?vc:gc)(e,1):this.index=e,this}setIndirect(e,n=0){return this.indirect=e,this.indirectOffset=n,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,n){return this.attributes[e]=n,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,n,i=0){this.groups.push({start:e,count:n,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,n){this.drawRange.start=e,this.drawRange.count=n}applyMatrix4(e){let n=this.attributes.position;n!==void 0&&(n.applyMatrix4(e),n.needsUpdate=!0);let i=this.attributes.normal;if(i!==void 0){let r=new Ne().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(e){return wi.makeRotationFromQuaternion(e),this.applyMatrix4(wi),this}rotateX(e){return wi.makeRotationX(e),this.applyMatrix4(wi),this}rotateY(e){return wi.makeRotationY(e),this.applyMatrix4(wi),this}rotateZ(e){return wi.makeRotationZ(e),this.applyMatrix4(wi),this}translate(e,n,i){return wi.makeTranslation(e,n,i),this.applyMatrix4(wi),this}scale(e,n,i){return wi.makeScale(e,n,i),this.applyMatrix4(wi),this}lookAt(e){return Xg.lookAt(e),Xg.updateMatrix(),this.applyMatrix4(Xg.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Po).negate(),this.translate(Po.x,Po.y,Po.z),this}setFromPoints(e){let n=this.getAttribute("position");if(n===void 0){let i=[];for(let s=0,r=e.length;s<r;s++){let a=e[s];i.push(a.x,a.y,a.z||0)}this.setAttribute("position",new Ri(i,3))}else{let i=Math.min(e.length,n.count);for(let s=0;s<i;s++){let r=e[s];n.setXYZ(s,r.x,r.y,r.z||0)}e.length>n.count&&De("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),n.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Nr);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Be("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new G(-1/0,-1/0,-1/0),new G(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),n)for(let i=0,s=n.length;i<s;i++){let r=n[i];oi.setFromBufferAttribute(r),this.morphTargetsRelative?(un.addVectors(this.boundingBox.min,oi.min),this.boundingBox.expandByPoint(un),un.addVectors(this.boundingBox.max,oi.max),this.boundingBox.expandByPoint(un)):(this.boundingBox.expandByPoint(oi.min),this.boundingBox.expandByPoint(oi.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Be('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Yo);let e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){Be("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new G,1/0);return}if(e){let i=this.boundingSphere.center;if(oi.setFromBufferAttribute(e),n)for(let r=0,a=n.length;r<a;r++){let o=n[r];lc.setFromBufferAttribute(o),this.morphTargetsRelative?(un.addVectors(oi.min,lc.min),oi.expandByPoint(un),un.addVectors(oi.max,lc.max),oi.expandByPoint(un)):(oi.expandByPoint(lc.min),oi.expandByPoint(lc.max))}oi.getCenter(i);let s=0;for(let r=0,a=e.count;r<a;r++)un.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(un));if(n)for(let r=0,a=n.length;r<a;r++){let o=n[r],l=this.morphTargetsRelative;for(let c=0,h=o.count;c<h;c++)un.fromBufferAttribute(o,c),l&&(Po.fromBufferAttribute(e,c),un.add(Po)),s=Math.max(s,i.distanceToSquared(un))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Be('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let e=this.index,n=this.attributes;if(e===null||n.position===void 0||n.normal===void 0||n.uv===void 0){Be("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let i=n.position,s=n.normal,r=n.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==i.count)&&(a=new li(new Float32Array(4*i.count),4),this.setAttribute("tangent",a));let o=[],l=[],c=new Uint8Array(i.count);for(let N=0;N<i.count;N++)o[N]=new G,l[N]=new G;let h=new G,p=new G,u=new G,d=new Oe,x=new Oe,A=new Oe,g=new G,f=new G;function m(N,C,B){c[N]=c[C]=c[B]=1,h.fromBufferAttribute(i,N),p.fromBufferAttribute(i,C),u.fromBufferAttribute(i,B),d.fromBufferAttribute(r,N),x.fromBufferAttribute(r,C),A.fromBufferAttribute(r,B),p.sub(h),u.sub(h),x.sub(d),A.sub(d);let F=1/(x.x*A.y-A.x*x.y);isFinite(F)&&(g.copy(p).multiplyScalar(A.y).addScaledVector(u,-x.y).multiplyScalar(F),f.copy(u).multiplyScalar(x.x).addScaledVector(p,-A.x).multiplyScalar(F),o[N].add(g),o[C].add(g),o[B].add(g),l[N].add(f),l[C].add(f),l[B].add(f))}let y=this.groups;y.length===0&&(y=[{start:0,count:e.count}]);for(let N=0,C=y.length;N<C;++N){let B=y[N],F=B.start,X=B.count;for(let D=F,V=F+X;D<V;D+=3)m(e.getX(D+0),e.getX(D+1),e.getX(D+2))}let _=new G,T=new G,b=new G,v=new G;function E(N){b.fromBufferAttribute(s,N),v.copy(b);let C=o[N];_.copy(C),_.sub(b.multiplyScalar(b.dot(C))).normalize(),T.crossVectors(v,C);let F=T.dot(l[N])<0?-1:1;a.setXYZW(N,_.x,_.y,_.z,F)}for(let N=0,C=i.count;N<C;++N)c[N]&&E(N);this._transformed=!0}computeVertexNormals(){let e=this.index,n=this.getAttribute("position");if(n!==void 0){let i=this.getAttribute("normal");if(i===void 0||i.count!==n.count)i=new li(new Float32Array(n.count*3),3),this.setAttribute("normal",i);else for(let u=0,d=i.count;u<d;u++)i.setXYZ(u,0,0,0);let s=new G,r=new G,a=new G,o=new G,l=new G,c=new G,h=new G,p=new G;if(e)for(let u=0,d=e.count;u<d;u+=3){let x=e.getX(u+0),A=e.getX(u+1),g=e.getX(u+2);s.fromBufferAttribute(n,x),r.fromBufferAttribute(n,A),a.fromBufferAttribute(n,g),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),o.fromBufferAttribute(i,x),l.fromBufferAttribute(i,A),c.fromBufferAttribute(i,g),o.add(h),l.add(h),c.add(h),i.setXYZ(x,o.x,o.y,o.z),i.setXYZ(A,l.x,l.y,l.z),i.setXYZ(g,c.x,c.y,c.z)}else for(let u=0,d=n.count;u<d;u+=3)s.fromBufferAttribute(n,u+0),r.fromBufferAttribute(n,u+1),a.fromBufferAttribute(n,u+2),h.subVectors(a,r),p.subVectors(s,r),h.cross(p),i.setXYZ(u+0,h.x,h.y,h.z),i.setXYZ(u+1,h.x,h.y,h.z),i.setXYZ(u+2,h.x,h.y,h.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){let e=this.attributes.normal;for(let n=0,i=e.count;n<i;n++)un.fromBufferAttribute(e,n),un.normalize(),e.setXYZ(n,un.x,un.y,un.z)}toNonIndexed(){function e(o,l){let c=o.array,h=o.itemSize,p=o.normalized,u=new c.constructor(l.length*h),d=0,x=0;for(let A=0,g=l.length;A<g;A++){o.isInterleavedBufferAttribute?d=l[A]*o.data.stride+o.offset:d=l[A]*h;for(let f=0;f<h;f++)u[x++]=c[d++]}return new li(u,h,p)}if(this.index===null)return De("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let n=new t,i=this.index.array,s=this.attributes;for(let o in s){let l=s[o],c=e(l,i);n.setAttribute(o,c)}let r=this.morphAttributes;for(let o in r){let l=[],c=r[o];for(let h=0,p=c.length;h<p;h++){let u=c[h],d=e(u,i);l.push(d)}n.morphAttributes[o]=l}n.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,l=a.length;o<l;o++){let c=a[o];n.addGroup(c.start,c.count,c.materialIndex)}return n}toJSON(){let e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,e.name=this.name,Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let l=this.parameters;for(let c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};let n=this.index;n!==null&&(e.data.index={type:n.array.constructor.name,array:Array.prototype.slice.call(n.array)});let i=this.attributes;for(let l in i){let c=i[l];e.data.attributes[l]=c.toJSON(e.data)}let s={},r=!1;for(let l in this.morphAttributes){let c=this.morphAttributes[l],h=[];for(let p=0,u=c.length;p<u;p++){let d=c[p];h.push(d.toJSON(e.data))}h.length>0&&(s[l]=h,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let n={};this.name=e.name;let i=e.index;i!==null&&this.setIndex(i.clone());let s=e.attributes;for(let c in s){let h=s[c];this.setAttribute(c,h.clone(n))}let r=e.morphAttributes;for(let c in r){let h=[],p=r[c];for(let u=0,d=p.length;u<d;u++)h.push(p[u].clone(n));this.morphAttributes[c]=h}this.morphTargetsRelative=e.morphTargetsRelative;let a=e.groups;for(let c=0,h=a.length;c<h;c++){let p=a[c];this.addGroup(p.start,p.count,p.materialIndex)}let o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());let l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this._transformed=e._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}};var Yg=new G,Bw=new G,Nw=new Ne,Gi=class{constructor(e=new G(1,0,0),n=0){this.isPlane=!0,this.normal=e,this.constant=n}set(e,n){return this.normal.copy(e),this.constant=n,this}setComponents(e,n,i,s){return this.normal.set(e,n,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,n){return this.normal.copy(e),this.constant=-n.dot(this.normal),this}setFromCoplanarPoints(e,n,i){let s=Yg.subVectors(i,n).cross(Bw.subVectors(e,n)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){let e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,n){return n.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,n,i=!0){let s=e.delta(Yg),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(e.start)===0?n.copy(e.start):null;let a=-(e.start.dot(this.normal)+this.constant)/r;return i===!0&&(a<0||a>1)?null:n.copy(e.start).addScaledVector(s,a)}intersectsLine(e){let n=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return n<0&&i>0||i<0&&n>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,n){let i=n||Nw.getNormalMatrix(e),s=this.coplanarPoint(Yg).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}toJSON(){return{normal:this.normal.toArray(),constant:this.constant}}fromJSON(e){return this.normal.fromArray(e.normal),this.constant=e.constant,this}},Lw=0,Vi=class extends qn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Lw++}),this.uuid=Bc(),this.name="",this.type="Material",this.blending=Ko,this.side=ms,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=rv,this.blendDst=av,this.blendEquation=Ma,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Ye(0,0,0),this.blendAlpha=0,this.depthFunc=Vo,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=bM,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Zf,this.stencilZFail=Zf,this.stencilZPass=Zf,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(let n in e){let i=e[n];if(i===void 0){De(`Material: parameter '${n}' has value of undefined.`);continue}let s=this[n];if(s===void 0){De(`Material: '${n}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector2&&i&&i.isVector2||s&&s.isEuler&&i&&i.isEuler||s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[n]=i}}toJSON(e){let n=e===void 0||typeof e=="string";n&&(e={textures:{},images:{}});let i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,i.blending=this.blending,i.side=this.side,i.shadowSide=this.shadowSide,i.vertexColors=this.vertexColors,i.opacity=this.opacity,i.transparent=this.transparent,i.blendSrc=this.blendSrc,i.blendDst=this.blendDst,i.blendEquation=this.blendEquation,i.blendSrcAlpha=this.blendSrcAlpha,i.blendDstAlpha=this.blendDstAlpha,i.blendEquationAlpha=this.blendEquationAlpha,i.blendColor=this.blendColor.getHex(),i.blendAlpha=this.blendAlpha,i.depthFunc=this.depthFunc,i.depthTest=this.depthTest,i.depthWrite=this.depthWrite,i.colorWrite=this.colorWrite,i.clipIntersection=this.clipIntersection,i.clipShadows=this.clipShadows,i.stencilWriteMask=this.stencilWriteMask,i.stencilFunc=this.stencilFunc,i.stencilRef=this.stencilRef,i.stencilFuncMask=this.stencilFuncMask,i.stencilFail=this.stencilFail,i.stencilZFail=this.stencilZFail,i.stencilZPass=this.stencilZPass,i.stencilWrite=this.stencilWrite,i.polygonOffset=this.polygonOffset,i.polygonOffsetFactor=this.polygonOffsetFactor,i.polygonOffsetUnits=this.polygonOffsetUnits,i.dithering=this.dithering,i.alphaTest=this.alphaTest,i.alphaHash=this.alphaHash,i.alphaToCoverage=this.alphaToCoverage,i.premultipliedAlpha=this.premultipliedAlpha,i.forceSinglePass=this.forceSinglePass,i.allowOverride=this.allowOverride,i.visible=this.visible,i.toneMapped=this.toneMapped,i.name=this.name,this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.diffuseRoughness!==void 0&&(i.diffuseRoughness=this.diffuseRoughness),this.diffuseRoughnessMap&&this.diffuseRoughnessMap.isTexture&&(i.diffuseRoughnessMap=this.diffuseRoughnessMap.toJSON(e).uuid),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.retroreflectivity!==void 0&&(i.retroreflectivity=this.retroreflectivity),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessMinimum!==void 0&&(i.iridescenceThicknessMinimum=this.iridescenceThicknessMinimum),this.iridescenceThicknessMaximum!==void 0&&(i.iridescenceThicknessMaximum=this.iridescenceThicknessMaximum),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),Array.isArray(this.clippingPlanes)&&this.clippingPlanes.length>0&&(i.clippingPlanes=this.clippingPlanes.map(r=>r.toJSON())),this.rotation!==void 0&&(i.rotation=this.rotation),this.depthPacking!==void 0&&(i.depthPacking=this.depthPacking),this.linewidth!==void 0&&(i.linewidth=this.linewidth),this.linecap!==void 0&&(i.linecap=this.linecap),this.linejoin!==void 0&&(i.linejoin=this.linejoin),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.wireframe!==void 0&&(i.wireframe=this.wireframe),this.wireframeLinewidth!==void 0&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!==void 0&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!==void 0&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading!==void 0&&(i.flatShading=this.flatShading),this.fog!==void 0&&(i.fog=this.fog),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){let a=[];for(let o in r){let l=r[o];delete l.metadata,a.push(l)}return a}if(n){let r=s(e.textures),a=s(e.images);r.length>0&&(i.textures=r),a.length>0&&(i.images=a)}return i}fromJSON(e,n){if(e.uuid!==void 0&&(this.uuid=e.uuid),e.name!==void 0&&(this.name=e.name),e.color!==void 0&&this.color!==void 0&&this.color.setHex(e.color),e.roughness!==void 0&&(this.roughness=e.roughness),e.metalness!==void 0&&(this.metalness=e.metalness),e.sheen!==void 0&&(this.sheen=e.sheen),e.sheenColor!==void 0&&(this.sheenColor=new Ye().setHex(e.sheenColor)),e.sheenRoughness!==void 0&&(this.sheenRoughness=e.sheenRoughness),e.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(e.emissive),e.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(e.specular),e.specularIntensity!==void 0&&(this.specularIntensity=e.specularIntensity),e.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(e.specularColor),e.shininess!==void 0&&(this.shininess=e.shininess),e.clearcoat!==void 0&&(this.clearcoat=e.clearcoat),e.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=e.clearcoatRoughness),e.diffuseRoughness!==void 0&&(this.diffuseRoughness=e.diffuseRoughness),e.dispersion!==void 0&&(this.dispersion=e.dispersion),e.retroreflectivity!==void 0&&(this.retroreflectivity=e.retroreflectivity),e.iridescence!==void 0&&(this.iridescence=e.iridescence),e.iridescenceIOR!==void 0&&(this.iridescenceIOR=e.iridescenceIOR),e.iridescenceThicknessMinimum!==void 0&&(this.iridescenceThicknessMinimum=e.iridescenceThicknessMinimum),e.iridescenceThicknessMaximum!==void 0&&(this.iridescenceThicknessMaximum=e.iridescenceThicknessMaximum),e.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessMinimum=e.iridescenceThicknessRange[0],this.iridescenceThicknessMaximum=e.iridescenceThicknessRange[1]),e.transmission!==void 0&&(this.transmission=e.transmission),e.thickness!==void 0&&(this.thickness=e.thickness),e.attenuationDistance!==void 0&&(this.attenuationDistance=e.attenuationDistance),e.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(e.attenuationColor),e.anisotropy!==void 0&&(this.anisotropy=e.anisotropy),e.anisotropyRotation!==void 0&&(this.anisotropyRotation=e.anisotropyRotation),e.fog!==void 0&&(this.fog=e.fog),e.flatShading!==void 0&&(this.flatShading=e.flatShading),e.blending!==void 0&&(this.blending=e.blending),e.combine!==void 0&&(this.combine=e.combine),e.side!==void 0&&(this.side=e.side),e.shadowSide!==void 0&&(this.shadowSide=e.shadowSide),e.opacity!==void 0&&(this.opacity=e.opacity),e.transparent!==void 0&&(this.transparent=e.transparent),e.alphaTest!==void 0&&(this.alphaTest=e.alphaTest),e.alphaHash!==void 0&&(this.alphaHash=e.alphaHash),e.depthFunc!==void 0&&(this.depthFunc=e.depthFunc),e.depthTest!==void 0&&(this.depthTest=e.depthTest),e.depthWrite!==void 0&&(this.depthWrite=e.depthWrite),e.colorWrite!==void 0&&(this.colorWrite=e.colorWrite),e.clippingPlanes!==void 0&&(this.clippingPlanes=e.clippingPlanes.map(i=>new Gi().fromJSON(i))),e.clipIntersection!==void 0&&(this.clipIntersection=e.clipIntersection),e.clipShadows!==void 0&&(this.clipShadows=e.clipShadows),e.depthPacking!==void 0&&(this.depthPacking=e.depthPacking),e.blendSrc!==void 0&&(this.blendSrc=e.blendSrc),e.blendDst!==void 0&&(this.blendDst=e.blendDst),e.blendEquation!==void 0&&(this.blendEquation=e.blendEquation),e.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=e.blendSrcAlpha),e.blendDstAlpha!==void 0&&(this.blendDstAlpha=e.blendDstAlpha),e.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=e.blendEquationAlpha),e.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(e.blendColor),e.blendAlpha!==void 0&&(this.blendAlpha=e.blendAlpha),e.stencilWriteMask!==void 0&&(this.stencilWriteMask=e.stencilWriteMask),e.stencilFunc!==void 0&&(this.stencilFunc=e.stencilFunc),e.stencilRef!==void 0&&(this.stencilRef=e.stencilRef),e.stencilFuncMask!==void 0&&(this.stencilFuncMask=e.stencilFuncMask),e.stencilFail!==void 0&&(this.stencilFail=e.stencilFail),e.stencilZFail!==void 0&&(this.stencilZFail=e.stencilZFail),e.stencilZPass!==void 0&&(this.stencilZPass=e.stencilZPass),e.stencilWrite!==void 0&&(this.stencilWrite=e.stencilWrite),e.wireframe!==void 0&&(this.wireframe=e.wireframe),e.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=e.wireframeLinewidth),e.wireframeLinecap!==void 0&&(this.wireframeLinecap=e.wireframeLinecap),e.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=e.wireframeLinejoin),e.rotation!==void 0&&(this.rotation=e.rotation),e.linewidth!==void 0&&(this.linewidth=e.linewidth),e.linecap!==void 0&&(this.linecap=e.linecap),e.linejoin!==void 0&&(this.linejoin=e.linejoin),e.dashSize!==void 0&&(this.dashSize=e.dashSize),e.gapSize!==void 0&&(this.gapSize=e.gapSize),e.scale!==void 0&&(this.scale=e.scale),e.polygonOffset!==void 0&&(this.polygonOffset=e.polygonOffset),e.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=e.polygonOffsetFactor),e.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=e.polygonOffsetUnits),e.dithering!==void 0&&(this.dithering=e.dithering),e.alphaToCoverage!==void 0&&(this.alphaToCoverage=e.alphaToCoverage),e.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=e.premultipliedAlpha),e.forceSinglePass!==void 0&&(this.forceSinglePass=e.forceSinglePass),e.allowOverride!==void 0&&(this.allowOverride=e.allowOverride),e.visible!==void 0&&(this.visible=e.visible),e.toneMapped!==void 0&&(this.toneMapped=e.toneMapped),e.userData!==void 0&&(this.userData=e.userData),e.vertexColors!==void 0&&(typeof e.vertexColors=="number"?this.vertexColors=e.vertexColors>0:this.vertexColors=e.vertexColors),e.size!==void 0&&(this.size=e.size),e.sizeAttenuation!==void 0&&(this.sizeAttenuation=e.sizeAttenuation),e.map!==void 0&&(this.map=n[e.map]||null),e.matcap!==void 0&&(this.matcap=n[e.matcap]||null),e.alphaMap!==void 0&&(this.alphaMap=n[e.alphaMap]||null),e.bumpMap!==void 0&&(this.bumpMap=n[e.bumpMap]||null),e.bumpScale!==void 0&&(this.bumpScale=e.bumpScale),e.normalMap!==void 0&&(this.normalMap=n[e.normalMap]||null),e.normalMapType!==void 0&&(this.normalMapType=e.normalMapType),e.normalScale!==void 0){let i=e.normalScale;Array.isArray(i)===!1&&(i=[i,i]),this.normalScale=new Oe().fromArray(i)}return e.displacementMap!==void 0&&(this.displacementMap=n[e.displacementMap]||null),e.displacementScale!==void 0&&(this.displacementScale=e.displacementScale),e.displacementBias!==void 0&&(this.displacementBias=e.displacementBias),e.roughnessMap!==void 0&&(this.roughnessMap=n[e.roughnessMap]||null),e.metalnessMap!==void 0&&(this.metalnessMap=n[e.metalnessMap]||null),e.emissiveMap!==void 0&&(this.emissiveMap=n[e.emissiveMap]||null),e.emissiveIntensity!==void 0&&(this.emissiveIntensity=e.emissiveIntensity),e.specularMap!==void 0&&(this.specularMap=n[e.specularMap]||null),e.specularIntensityMap!==void 0&&(this.specularIntensityMap=n[e.specularIntensityMap]||null),e.specularColorMap!==void 0&&(this.specularColorMap=n[e.specularColorMap]||null),e.envMap!==void 0&&(this.envMap=n[e.envMap]||null),e.envMapRotation!==void 0&&this.envMapRotation.fromArray(e.envMapRotation),e.envMapIntensity!==void 0&&(this.envMapIntensity=e.envMapIntensity),e.reflectivity!==void 0&&(this.reflectivity=e.reflectivity),e.refractionRatio!==void 0&&(this.refractionRatio=e.refractionRatio),e.lightMap!==void 0&&(this.lightMap=n[e.lightMap]||null),e.lightMapIntensity!==void 0&&(this.lightMapIntensity=e.lightMapIntensity),e.aoMap!==void 0&&(this.aoMap=n[e.aoMap]||null),e.aoMapIntensity!==void 0&&(this.aoMapIntensity=e.aoMapIntensity),e.gradientMap!==void 0&&(this.gradientMap=n[e.gradientMap]||null),e.clearcoatMap!==void 0&&(this.clearcoatMap=n[e.clearcoatMap]||null),e.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=n[e.clearcoatRoughnessMap]||null),e.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=n[e.clearcoatNormalMap]||null),e.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new Oe().fromArray(e.clearcoatNormalScale)),e.diffuseRoughnessMap!==void 0&&(this.diffuseRoughnessMap=n[e.diffuseRoughnessMap]||null),e.iridescenceMap!==void 0&&(this.iridescenceMap=n[e.iridescenceMap]||null),e.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=n[e.iridescenceThicknessMap]||null),e.transmissionMap!==void 0&&(this.transmissionMap=n[e.transmissionMap]||null),e.thicknessMap!==void 0&&(this.thicknessMap=n[e.thicknessMap]||null),e.anisotropyMap!==void 0&&(this.anisotropyMap=n[e.anisotropyMap]||null),e.sheenColorMap!==void 0&&(this.sheenColorMap=n[e.sheenColorMap]||null),e.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=n[e.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;let n=e.clippingPlanes,i=null;if(n!==null){let s=n.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=n[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}};var Gs=new G,qg=new G,Lf=new G,Pf=new G,ch=class{constructor(e=new G,n=new G(0,0,-1)){this.origin=e,this.direction=n}set(e,n){return this.origin.copy(e),this.direction.copy(n),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,n){return n.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Gs)),this}closestPointToPoint(e,n){n.subVectors(e,this.origin);let i=n.dot(this.direction);return i<0?n.copy(this.origin):n.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){let n=Gs.subVectors(e,this.origin).dot(this.direction);return n<0?this.origin.distanceToSquared(e):(Gs.copy(this.origin).addScaledVector(this.direction,n),Gs.distanceToSquared(e))}distanceSqToSegment(e,n,i,s){qg.copy(e).add(n).multiplyScalar(.5),Lf.copy(n).sub(e).normalize(),Pf.copy(this.origin).sub(qg);let r=e.distanceTo(n)*.5,a=-this.direction.dot(Lf),o=Pf.dot(this.direction),l=-Pf.dot(Lf),c=Pf.lengthSq(),h=Math.abs(1-a*a),p,u,d,x;if(h>0)if(p=a*l-o,u=a*o-l,x=r*h,p>=0)if(u>=-x)if(u<=x){let A=1/h;p*=A,u*=A,d=p*(p+a*u+2*o)+u*(a*p+u+2*l)+c}else u=r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u=-r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;else u<=-x?(p=Math.max(0,-(-a*r+o)),u=p>0?-r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c):u<=x?(p=0,u=Math.min(Math.max(-r,-l),r),d=u*(u+2*l)+c):(p=Math.max(0,-(a*r+o)),u=p>0?r:Math.min(Math.max(-r,-l),r),d=-p*p+u*(u+2*l)+c);else u=a>0?-r:r,p=Math.max(0,-(a*u+o)),d=-p*p+u*(u+2*l)+c;return i&&i.copy(this.origin).addScaledVector(this.direction,p),s&&s.copy(qg).addScaledVector(Lf,u),d}intersectSphere(e,n){if(e.radius<0)return null;Gs.subVectors(e.center,this.origin);let i=Gs.dot(this.direction),s=Gs.dot(Gs)-i*i,r=e.radius*e.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=i-a,l=i+a;return l<0?null:o<0?this.at(l,n):this.at(o,n)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){let n=e.normal.dot(this.direction);if(n===0)return e.distanceToPoint(this.origin)===0?0:null;let i=-(this.origin.dot(e.normal)+e.constant)/n;return i>=0?i:null}intersectPlane(e,n){let i=this.distanceToPlane(e);return i===null?null:this.at(i,n)}intersectsPlane(e){let n=e.distanceToPoint(this.origin);return n===0||e.normal.dot(this.direction)*n<0}intersectBox(e,n){let i,s,r,a,o,l,c=1/this.direction.x,h=1/this.direction.y,p=1/this.direction.z,u=this.origin;return c>=0?(i=(e.min.x-u.x)*c,s=(e.max.x-u.x)*c):(i=(e.max.x-u.x)*c,s=(e.min.x-u.x)*c),h>=0?(r=(e.min.y-u.y)*h,a=(e.max.y-u.y)*h):(r=(e.max.y-u.y)*h,a=(e.min.y-u.y)*h),i>a||r>s||((r>i||isNaN(i))&&(i=r),(a<s||isNaN(s))&&(s=a),p>=0?(o=(e.min.z-u.z)*p,l=(e.max.z-u.z)*p):(o=(e.max.z-u.z)*p,l=(e.min.z-u.z)*p),i>l||o>s)||((o>i||i!==i)&&(i=o),(l<s||s!==s)&&(s=l),s<0)?null:this.at(i>=0?i:s,n)}intersectsBox(e){return this.intersectBox(e,Gs)!==null}intersectTriangle(e,n,i,s,r){let a=this.origin,o=this.direction,l=o.x,c=o.y,h=o.z,p=e.x-a.x,u=e.y-a.y,d=e.z-a.z,x=n.x-a.x,A=n.y-a.y,g=n.z-a.z,f=i.x-a.x,m=i.y-a.y,y=i.z-a.z,_=Math.abs(l),T=Math.abs(c),b=Math.abs(h),v,E,N,C,B,F,X,D,V,q,H,te;if(_>=T&&_>=b?(N=l,F=p,V=x,te=f,l>=0?(v=c,E=h,C=u,B=d,X=A,D=g,q=m,H=y):(v=h,E=c,C=d,B=u,X=g,D=A,q=y,H=m)):T>=b?(N=c,F=u,V=A,te=m,c>=0?(v=h,E=l,C=d,B=p,X=g,D=x,q=y,H=f):(v=l,E=h,C=p,B=d,X=x,D=g,q=f,H=y)):(N=h,F=d,V=g,te=y,h>=0?(v=l,E=c,C=p,B=u,X=x,D=A,q=f,H=m):(v=c,E=l,C=u,B=p,X=A,D=x,q=m,H=f)),N===0)return null;let k=v/N,$=E/N,ee=1/N,_e=C-k*F,Ie=B-$*F,Q=X-k*V,de=D-$*V,xe=q-k*te,Te=H-$*te,pe=xe*de-Te*Q,be=_e*Te-Ie*xe,qe=Q*Ie-de*_e;if(s){if(pe<0||be<0||qe<0)return null}else if((pe<0||be<0||qe<0)&&(pe>0||be>0||qe>0))return null;let Ce=pe+be+qe;if(Ce===0)return null;let st=ee*(pe*F+be*V+qe*te);return(Ce>0?st<0:st>0)?null:this.at(st/Ce,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},uh=class extends Vi{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Ye(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Ir,this.combine=ov,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}},YA=new Ft,xa=new ch,Of=new Yo,qA=new G,Ff=new G,zf=new G,Hf=new G,Qg=new G,Gf=new G,QA=new G,Vf=new G,Qn=class extends Di{constructor(e=new ps,n=new uh){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=n,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,n){return super.copy(e,n),e.morphTargetInfluences!==void 0?this.morphTargetInfluences=e.morphTargetInfluences.slice():this.morphTargetInfluences=void 0,e.morphTargetDictionary!==void 0?this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary):this.morphTargetDictionary=void 0,this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){let n=this.geometry.morphAttributes,i=Object.keys(n);if(i.length>0){let s=n[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(e,n){let i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,a=i.morphTargetsRelative;n.fromBufferAttribute(s,e);let o=this.morphTargetInfluences;if(r&&o){Gf.set(0,0,0);for(let l=0,c=r.length;l<c;l++){let h=o[l],p=r[l];h!==0&&(Qg.fromBufferAttribute(p,e),a?Gf.addScaledVector(Qg,h):Gf.addScaledVector(Qg.sub(n),h))}n.add(Gf)}return n}intersectsFrustum(e){return e.intersectsObject(this)}raycast(e,n){let i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),Of.copy(i.boundingSphere),Of.applyMatrix4(r),xa.copy(e.ray).recast(e.near),!(Of.containsPoint(xa.origin)===!1&&(xa.intersectSphere(Of,qA)===null||xa.origin.distanceToSquared(qA)>(e.far-e.near)**2))&&(YA.copy(r).invert(),xa.copy(e.ray).applyMatrix4(YA),!(i.boundingBox!==null&&xa.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,n,xa)))}_computeIntersections(e,n,i){let s,r=this.geometry,a=this.material,o=r.index,l=r.attributes.position,c=r.attributes.uv,h=r.attributes.uv1,p=r.attributes.normal,u=r.groups,d=r.drawRange;if(o!==null)if(Array.isArray(a))for(let x=0,A=u.length;x<A;x++){let g=u[x],f=a[g.materialIndex],m=Math.max(g.start,d.start),y=Math.min(o.count,Math.min(g.start+g.count,d.start+d.count));for(let _=m,T=y;_<T;_+=3){let b=o.getX(_),v=o.getX(_+1),E=o.getX(_+2);s=kf(this,f,e,i,c,h,p,b,v,E),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=g.materialIndex,n.push(s))}}else{let x=Math.max(0,d.start),A=Math.min(o.count,d.start+d.count);for(let g=x,f=A;g<f;g+=3){let m=o.getX(g),y=o.getX(g+1),_=o.getX(g+2);s=kf(this,a,e,i,c,h,p,m,y,_),s&&(s.faceIndex=Math.floor(g/3),n.push(s))}}else if(l!==void 0)if(Array.isArray(a))for(let x=0,A=u.length;x<A;x++){let g=u[x],f=a[g.materialIndex],m=Math.max(g.start,d.start),y=Math.min(l.count,Math.min(g.start+g.count,d.start+d.count));for(let _=m,T=y;_<T;_+=3){let b=_,v=_+1,E=_+2;s=kf(this,f,e,i,c,h,p,b,v,E),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=g.materialIndex,n.push(s))}}else{let x=Math.max(0,d.start),A=Math.min(l.count,d.start+d.count);for(let g=x,f=A;g<f;g+=3){let m=g,y=g+1,_=g+2;s=kf(this,a,e,i,c,h,p,m,y,_),s&&(s.faceIndex=Math.floor(g/3),n.push(s))}}}};function Pw(t,e,n,i,s,r,a,o){let l;if(e.side===Qt?l=i.intersectTriangle(a,r,s,!0,o):l=i.intersectTriangle(s,r,a,e.side===ms,o),l===null)return null;Vf.copy(o),Vf.applyMatrix4(t.matrixWorld);let c=n.ray.origin.distanceTo(Vf);return c<n.near||c>n.far?null:{distance:c,point:Vf.clone(),object:t}}function kf(t,e,n,i,s,r,a,o,l,c){t.getVertexPosition(o,Ff),t.getVertexPosition(l,zf),t.getVertexPosition(c,Hf);let h=Pw(t,e,n,i,Ff,zf,Hf,QA);if(h){let p=new G;Ur.getBarycoord(QA,Ff,zf,Hf,p),s&&(h.uv=Ur.getInterpolatedAttribute(s,o,l,c,p,new Oe)),r&&(h.uv1=Ur.getInterpolatedAttribute(r,o,l,c,p,new Oe)),a&&(h.normal=Ur.getInterpolatedAttribute(a,o,l,c,p,new G),h.normal.dot(i.direction)>0&&h.normal.multiplyScalar(-1));let u={a:o,b:l,c,normal:new G,materialIndex:0};Ur.getNormal(Ff,zf,Hf,u.normal),h.face=u,h.barycoord=p}return h}var fh=class extends qt{constructor(e=null,n=1,i=1,s,r,a,o,l,c=fn,h=fn,p,u){super(null,a,o,l,c,h,s,r,p,u),this.isDataTexture=!0,this.image={data:e,width:n,height:i},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var ya=new Yo,Ow=new Oe(.5,.5),Wf=new G,Sa=class{constructor(e=new Gi,n=new Gi,i=new Gi,s=new Gi,r=new Gi,a=new Gi){this.planes=[e,n,i,s,r,a]}set(e,n,i,s,r,a){let o=this.planes;return o[0].copy(e),o[1].copy(n),o[2].copy(i),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(e){let n=this.planes;for(let i=0;i<6;i++)n[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,n=Ci,i=!1){let s=this.planes,r=e.elements,a=r[0],o=r[1],l=r[2],c=r[3],h=r[4],p=r[5],u=r[6],d=r[7],x=r[8],A=r[9],g=r[10],f=r[11],m=r[12],y=r[13],_=r[14],T=r[15];if(s[0].setComponents(c-a,d-h,f-x,T-m).normalize(),s[1].setComponents(c+a,d+h,f+x,T+m).normalize(),s[2].setComponents(c+o,d+p,f+A,T+y).normalize(),s[3].setComponents(c-o,d-p,f-A,T-y).normalize(),i)s[4].setComponents(l,u,g,_).normalize(),s[5].setComponents(c-l,d-u,f-g,T-_).normalize();else if(s[4].setComponents(c-l,d-u,f-g,T-_).normalize(),n===Ci)s[5].setComponents(c+l,d+u,f+g,T+_).normalize();else if(n===hc)s[5].setComponents(l,u,g,_).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+n);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),ya.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{let n=e.geometry;n.boundingSphere===null&&n.computeBoundingSphere(),ya.copy(n.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(ya)}intersectsSprite(e){ya.center.set(0,0,0);let n=Ow.distanceTo(e.center);return ya.radius=.7071067811865476+n,ya.applyMatrix4(e.matrixWorld),this.intersectsSphere(ya)}intersectsSphere(e){let n=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(n[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){let n=this.planes;for(let i=0;i<6;i++){let s=n[i];if(Wf.x=s.normal.x>0?e.max.x:e.min.x,Wf.y=s.normal.y>0?e.max.y:e.min.y,Wf.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Wf)<0)return!1}return!0}containsPoint(e){let n=this.planes;for(let i=0;i<6;i++)if(n[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}},ZA=new Ft,hh=class t{constructor(){this.coordinateSystem=Ci,this._frustums=[],this._count=0}setFromArrayCamera(e){let n=e.cameras,i=this._frustums;for(let s=0;s<n.length;s++){let r=n[s];ZA.multiplyMatrices(r.projectionMatrix,r.matrixWorldInverse),i[s]===void 0&&(i[s]=new Sa),i[s].setFromProjectionMatrix(ZA,r.coordinateSystem,r.reversedDepth)}return this._count=n.length,this}getFrustum(e){return this._frustums[e]}intersectsObject(e){let n=this._frustums;for(let i=0;i<this._count;i++)if(n[i].intersectsObject(e))return!0;return!1}intersectsSprite(e){let n=this._frustums;for(let i=0;i<this._count;i++)if(n[i].intersectsSprite(e))return!0;return!1}intersectsSphere(e){let n=this._frustums;for(let i=0;i<this._count;i++)if(n[i].intersectsSphere(e))return!0;return!1}intersectsBox(e){let n=this._frustums;for(let i=0;i<this._count;i++)if(n[i].intersectsBox(e))return!0;return!1}containsPoint(e){let n=this._frustums;for(let i=0;i<this._count;i++)if(n[i].containsPoint(e))return!0;return!1}copy(e){this.coordinateSystem=e.coordinateSystem;let n=this._frustums,i=e._frustums;for(let s=0;s<e._count;s++)n[s]===void 0&&(n[s]=new Sa),n[s].copy(i[s]);return this._count=e._count,this}clone(){return new t().copy(this)}};var xc=class extends qt{constructor(e=[],n=Ea,i,s,r,a,o,l,c,h){super(e,n,i,s,r,a,o,l,c,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}};var ci=class extends qt{constructor(e,n,i=Wi,s,r,a,o=fn,l=fn,c,h=ks,p=1){if(h!==ks&&h!==qi)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let u={width:e,height:n,depth:p};super(u,s,r,a,o,l,h,i,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Wo(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){let n=super.toJSON(e);return n.compareFunction=this.compareFunction,n}},yc=class extends ci{constructor(e,n=Wi,i=Ea,s,r,a=fn,o=fn,l,c=ks){let h={width:e,height:e,depth:1},p=[h,h,h,h,h,h];super(e,e,n,i,s,r,a,o,l,c),this.image=p,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}},_c=class extends qt{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}},qo=class t extends ps{constructor(e=1,n=1,i=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:n,depth:i,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let l=[],c=[],h=[],p=[],u=0,d=0;x("z","y","x",-1,-1,i,n,e,a,r,0),x("z","y","x",1,-1,i,n,-e,a,r,1),x("x","z","y",1,1,e,i,n,s,a,2),x("x","z","y",1,-1,e,i,-n,s,a,3),x("x","y","z",1,-1,e,n,i,s,r,4),x("x","y","z",-1,-1,e,n,-i,s,r,5),this.setIndex(l),this.setAttribute("position",new Ri(c,3)),this.setAttribute("normal",new Ri(h,3)),this.setAttribute("uv",new Ri(p,2));function x(A,g,f,m,y,_,T,b,v,E,N){let C=_/v,B=T/E,F=_/2,X=T/2,D=b/2,V=v+1,q=E+1,H=0,te=0,k=new G;for(let $=0;$<q;$++){let ee=$*B-X;for(let _e=0;_e<V;_e++){let Ie=_e*C-F;k[A]=Ie*m,k[g]=ee*y,k[f]=D,c.push(k.x,k.y,k.z),k[A]=0,k[g]=0,k[f]=b>0?1:-1,h.push(k.x,k.y,k.z),p.push(_e/v),p.push(1-$/E),H+=1}}for(let $=0;$<E;$++)for(let ee=0;ee<v;ee++){let _e=u+ee+V*$,Ie=u+ee+V*($+1),Q=u+(ee+1)+V*($+1),de=u+(ee+1)+V*$;l.push(_e,Ie,de),l.push(Ie,Q,de),te+=6}o.addGroup(d,te,N),d+=te,u+=H}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};var Aa=class t extends ps{constructor(e=1,n=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:n,widthSegments:i,heightSegments:s};let r=e/2,a=n/2,o=Math.floor(i),l=Math.floor(s),c=o+1,h=l+1,p=e/o,u=n/l,d=[],x=[],A=[],g=[];for(let f=0;f<h;f++){let m=f*u-a;for(let y=0;y<c;y++){let _=y*p-r;x.push(_,-m,0),A.push(0,0,1),g.push(y/o),g.push(1-f/l)}}for(let f=0;f<l;f++)for(let m=0;m<o;m++){let y=m+c*f,_=m+c*(f+1),T=m+1+c*(f+1),b=m+1+c*f;d.push(y,_,b),d.push(_,T,b)}this.setIndex(d),this.setAttribute("position",new Ri(x,3)),this.setAttribute("normal",new Ri(A,3)),this.setAttribute("uv",new Ri(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new t(e.width,e.height,e.widthSegments,e.heightSegments)}};function Ta(t){let e={};for(let n in t){e[n]={};for(let i in t[n]){let s=t[n][i];if(KA(s))s.isRenderTargetTexture?(De("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[n][i]=null):e[n][i]=s.clone();else if(Array.isArray(s))if(KA(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();e[n][i]=r}else e[n][i]=s.slice();else e[n][i]=s}}return e}function Rn(t){let e={};for(let n=0;n<t.length;n++){let i=Ta(t[n]);for(let s in i)e[s]=i[s]}return e}function KA(t){return t&&(t.isColor||t.isMatrix3||t.isMatrix4||t.isVector2||t.isVector3||t.isVector4||t.isTexture||t.isQuaternion)}function Fw(t){let e=[];for(let n=0;n<t.length;n++)e.push(t[n].clone());return e}function Rv(t){let e=t.getRenderTarget();return e===null?t.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:je.workingColorSpace}var zM={clone:Ta,merge:Rn},zw=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Hw=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,hn=class extends Vi{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=zw,this.fragmentShader=Hw,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Ta(e.uniforms),this.uniformsGroups=Fw(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){let n=super.toJSON(e);n.glslVersion=this.glslVersion,n.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?n.uniforms[s]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?n.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?n.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?n.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?n.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?n.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?n.uniforms[s]={type:"m4",value:a.toArray()}:n.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(n.defines=this.defines),n.vertexShader=this.vertexShader,n.fragmentShader=this.fragmentShader,n.lights=this.lights,n.clipping=this.clipping;let i={};for(let s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(n.extensions=i),n}fromJSON(e,n){if(super.fromJSON(e,n),e.uniforms!==void 0)for(let i in e.uniforms){let s=e.uniforms[i];switch(this.uniforms[i]={},s.type){case"t":this.uniforms[i].value=n[s.value]||null;break;case"c":this.uniforms[i].value=new Ye().setHex(s.value);break;case"v2":this.uniforms[i].value=new Oe().fromArray(s.value);break;case"v3":this.uniforms[i].value=new G().fromArray(s.value);break;case"v4":this.uniforms[i].value=new Lt().fromArray(s.value);break;case"m3":this.uniforms[i].value=new Ne().fromArray(s.value);break;case"m4":this.uniforms[i].value=new Ft().fromArray(s.value);break;default:this.uniforms[i].value=s.value}}if(e.defines!==void 0&&(this.defines=e.defines),e.vertexShader!==void 0&&(this.vertexShader=e.vertexShader),e.fragmentShader!==void 0&&(this.fragmentShader=e.fragmentShader),e.glslVersion!==void 0&&(this.glslVersion=e.glslVersion),e.extensions!==void 0)for(let i in e.extensions)this.extensions[i]=e.extensions[i];return e.lights!==void 0&&(this.lights=e.lights),e.clipping!==void 0&&(this.clipping=e.clipping),this}},dh=class extends hn{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}};var ph=class extends Vi{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=vs,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}},mh=class extends Vi{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}};function Oo(t,e){return!t||t.constructor===e?t:typeof e.BYTES_PER_ELEMENT=="number"?new e(t):Array.prototype.slice.call(t)}function Xf(t){return t!==void 0&&t.inTangents!==void 0&&t.outTangents!==void 0}var Lr=class{constructor(e,n,i,s){this.parameterPositions=e,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new n.constructor(i),this.sampleValues=n,this.valueSize=i,this.settings=null,this.DefaultSettings_={}}evaluate(e){let n=this.parameterPositions,i=this._cachedIndex,s=n[i],r=n[i-1];e:{t:{let a;n:{i:if(!(e<s)){for(let o=i+2;;){if(s===void 0){if(e<r)break i;return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}if(i===o)break;if(r=s,s=n[++i],e<s)break t}a=n.length;break n}if(!(e>=r)){let o=n[1];e<o&&(i=2,r=o);for(let l=i-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===l)break;if(s=r,r=n[--i-1],e>=r)break t}a=i,i=0;break n}break e}for(;i<a;){let o=i+a>>>1;e<n[o]?a=o:i=o+1}if(s=n[i],r=n[i-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return i=n.length,this._cachedIndex=i,this.copySampleValue_(i-1)}this._cachedIndex=i,this.intervalChanged_(i,r,s)}return this.interpolate_(i,r,e,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(e){let n=this.resultBuffer,i=this.sampleValues,s=this.valueSize,r=e*s;for(let a=0;a!==s;++a)n[a]=i[r+a];return n}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},gh=class extends Lr{constructor(e,n,i,s){super(e,n,i,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:Kg,endingEnd:Kg}}intervalChanged_(e,n,i){let s=this.parameterPositions,r=e-2,a=e+1,o=s[r],l=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case Jg:r=e,o=2*n-i;break;case jg:r=s.length-2,o=n+s[r]-s[r+1];break;default:r=e,o=i}if(l===void 0)switch(this.getSettings_().endingEnd){case Jg:a=e,l=2*i-n;break;case jg:a=1,l=i+s[1]-s[0];break;default:a=e-1,l=n}let c=(i-n)*.5,h=this.valueSize;this._weightPrev=c/(n-o),this._weightNext=c/(l-i),this._offsetPrev=r*h,this._offsetNext=a*h}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this._offsetPrev,p=this._offsetNext,u=this._weightPrev,d=this._weightNext,x=(i-n)/(s-n),A=x*x,g=A*x,f=-u*g+2*u*A-u*x,m=(1+u)*g+(-1.5-2*u)*A+(-.5+u)*x+1,y=(-1-d)*g+(1.5+d)*A+.5*x,_=d*g-d*A;for(let T=0;T!==o;++T)r[T]=f*a[h+T]+m*a[c+T]+y*a[l+T]+_*a[p+T];return r}},vh=class extends Lr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=(i-n)/(s-n),p=1-h;for(let u=0;u!==o;++u)r[u]=a[c+u]*p+a[l+u]*h;return r}},xh=class extends Lr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e){return this.copySampleValue_(e-1)}},yh=class extends Lr{interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=e*o,c=l-o,h=this.inTangents,p=this.outTangents;if(!h||!p){let x=(i-n)/(s-n),A=1-x;for(let g=0;g!==o;++g)r[g]=a[c+g]*A+a[l+g]*x;return r}let u=o*2,d=e-1;for(let x=0;x!==o;++x){let A=a[c+x],g=a[l+x],f=d*u+x*2,m=p[f],y=p[f+1],_=e*u+x*2,T=h[_],b=h[_+1],v=Vw(i,n,m,T,s);r[x]=HM(v,A,y,b,g)}return r}};function HM(t,e,n,i,s){let r=1-t;return r*r*r*e+3*r*r*t*n+3*r*t*t*i+t*t*t*s}function Gw(t,e,n,i,s){let r=1-t;return 3*r*r*(n-e)+6*r*t*(i-n)+3*t*t*(s-i)}function Vw(t,e,n,i,s){let r=(t-e)/(s-e);for(let a=0;a<8;a++){let o=HM(r,e,n,i,s)-t;if(Math.abs(o)<1e-10)break;let l=Gw(r,e,n,i,s);if(Math.abs(l)<1e-10)break;r=Math.max(0,Math.min(1,r-o/l))}return r}var ui=class{constructor(e,n,i,s){if(e===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(n===void 0||n.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+e);this.name=e,this.times=Oo(n,this.TimeBufferType),this.values=Oo(i,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(e){let n=e.constructor,i;if(n.toJSON!==this.toJSON)i=n.toJSON(e);else{i={name:e.name,times:Oo(e.times,Array),values:Oo(e.values,Array)};let s=e.getInterpolation();s!==e.DefaultInterpolation&&(i.interpolation=s),Xf(e.settings)&&(i.settings={inTangents:Oo(e.settings.inTangents,Array),outTangents:Oo(e.settings.outTangents,Array)})}return i.type=e.ValueTypeName,i}InterpolantFactoryMethodDiscrete(e){return new xh(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodLinear(e){return new vh(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodSmooth(e){return new gh(this.times,this.values,this.getValueSize(),e)}InterpolantFactoryMethodBezier(e){let n=new yh(this.times,this.values,this.getValueSize(),e);return this.settings&&(n.inTangents=this.settings.inTangents,n.outTangents=this.settings.outTangents),n}setInterpolation(e){let n;switch(e){case uc:n=this.InterpolantFactoryMethodDiscrete;break;case sh:n=this.InterpolantFactoryMethodLinear;break;case Qf:n=this.InterpolantFactoryMethodSmooth;break;case Zg:n=this.InterpolantFactoryMethodBezier;break}if(n===void 0){let i="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(e!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(i);return De("KeyframeTrack:",i),this}return this.createInterpolant=n,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return uc;case this.InterpolantFactoryMethodLinear:return sh;case this.InterpolantFactoryMethodSmooth:return Qf;case this.InterpolantFactoryMethodBezier:return Zg}}getValueSize(){return this.values.length/this.times.length}shift(e){if(e!==0){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]+=e;Xf(this.settings)&&(JA(this.settings.inTangents,e),JA(this.settings.outTangents,e))}return this}scale(e){if(e!==1){let n=this.times;for(let i=0,s=n.length;i!==s;++i)n[i]*=e;Xf(this.settings)&&(jA(this.settings.inTangents,e),jA(this.settings.outTangents,e))}return this}trim(e,n){let i=this.times,s=i.length,r=0,a=s-1;for(;r!==s&&i[r]<e;)++r;for(;a!==-1&&i[a]>n;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=i.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let e=!0,n=this.getValueSize();n-Math.floor(n)!==0&&(Be("KeyframeTrack: Invalid value size in track.",this),e=!1);let i=this.times,s=this.values,r=i.length;r===0&&(Be("KeyframeTrack: Track is empty.",this),e=!1);let a=null;for(let o=0;o!==r;o++){let l=i[o];if(typeof l=="number"&&isNaN(l)){Be("KeyframeTrack: Time is not a valid number.",this,o,l),e=!1;break}if(a!==null&&a>l){Be("KeyframeTrack: Out of order keys.",this,o,l,a),e=!1;break}a=l}if(s!==void 0&&gw(s))for(let o=0,l=s.length;o!==l;++o){let c=s[o];if(isNaN(c)){Be("KeyframeTrack: Value is not a valid number.",this,o,c),e=!1;break}}return e}optimize(){let e=this.times.slice(),n=this.values.slice(),i=this.getValueSize(),s=this.getInterpolation()===Qf,r=e.length-1,a=1;for(let o=1;o<r;++o){let l=!1,c=e[o],h=e[o+1];if(c!==h&&(o!==1||c!==e[0]))if(s)l=!0;else{let p=o*i,u=p-i,d=p+i;for(let x=0;x!==i;++x){let A=n[p+x];if(A!==n[u+x]||A!==n[d+x]){l=!0;break}}}if(l){if(o!==a){e[a]=e[o];let p=o*i,u=a*i;for(let d=0;d!==i;++d)n[u+d]=n[p+d]}++a}}if(r>0){e[a]=e[r];for(let o=r*i,l=a*i,c=0;c!==i;++c)n[l+c]=n[o+c];++a}return a!==e.length?(this.times=e.slice(0,a),this.values=n.slice(0,a*i)):(this.times=e,this.values=n),this}clone(){let e=this.times.slice(),n=this.values.slice(),i=this.constructor,s=new i(this.name,e,n);return s.createInterpolant=this.createInterpolant,Xf(this.settings)&&(s.settings={inTangents:this.settings.inTangents.slice(),outTangents:this.settings.outTangents.slice()}),s}};function JA(t,e){for(let n=0,i=t.length;n!==i;n+=2)t[n]+=e}function jA(t,e){for(let n=0,i=t.length;n!==i;n+=2)t[n]*=e}ui.prototype.ValueTypeName="";ui.prototype.TimeBufferType=Float32Array;ui.prototype.ValueBufferType=Float32Array;ui.prototype.DefaultInterpolation=sh;var Pr=class extends ui{constructor(e,n,i){super(e,n,i)}};Pr.prototype.ValueTypeName="bool";Pr.prototype.ValueBufferType=Array;Pr.prototype.DefaultInterpolation=uc;Pr.prototype.InterpolantFactoryMethodLinear=void 0;Pr.prototype.InterpolantFactoryMethodSmooth=void 0;var _h=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};_h.prototype.ValueTypeName="color";var Sh=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};Sh.prototype.ValueTypeName="number";var Ah=class extends Lr{constructor(e,n,i,s){super(e,n,i,s)}interpolate_(e,n,i,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,l=(i-n)/(s-n),c=e*o;for(let h=c+o;c!==h;c+=4)ds.slerpFlat(r,0,a,c-o,a,c,l);return r}},Sc=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}InterpolantFactoryMethodLinear(e){return new Ah(this.times,this.values,this.getValueSize(),e)}};Sc.prototype.ValueTypeName="quaternion";Sc.prototype.InterpolantFactoryMethodSmooth=void 0;var Or=class extends ui{constructor(e,n,i){super(e,n,i)}};Or.prototype.ValueTypeName="string";Or.prototype.ValueBufferType=Array;Or.prototype.DefaultInterpolation=uc;Or.prototype.InterpolantFactoryMethodLinear=void 0;Or.prototype.InterpolantFactoryMethodSmooth=void 0;var Mh=class extends ui{constructor(e,n,i,s){super(e,n,i,s)}};Mh.prototype.ValueTypeName="vector";var Eh=class{constructor(e,n,i){let s=this,r=!1,a=0,o=0,l,c=[];this.onStart=void 0,this.onLoad=e,this.onProgress=n,this.onError=i,this._abortController=null,this.itemStart=function(h){o++,r===!1&&s.onStart!==void 0&&s.onStart(h,a,o),r=!0},this.itemEnd=function(h){a++,s.onProgress!==void 0&&s.onProgress(h,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(h){s.onError!==void 0&&s.onError(h)},this.resolveURL=function(h){return h=h.normalize("NFC"),l?l(h):h},this.setURLModifier=function(h){return l=h,this},this.addHandler=function(h,p){return c.push(h,p),this},this.removeHandler=function(h){let p=c.indexOf(h);return p!==-1&&c.splice(p,2),this},this.getHandler=function(h){for(let p=0,u=c.length;p<u;p+=2){let d=c[p],x=c[p+1];if(d.global&&(d.lastIndex=0),d.test(h))return x}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},GM=new Eh,Th=class{constructor(e){this.manager=e!==void 0?e:GM,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},this.priority="auto",this.cache="default",typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,n){let i=this;return new Promise(function(s,r){i.load(e,s,n,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}setPriority(e){return this.priority=e,this}setCache(e){return this.cache=e,this}abort(){return this}};Th.DEFAULT_MATERIAL_NAME="__DEFAULT";var Yf=new G,qf=new ds,fs=new G,Ac=class extends Di{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Ft,this.projectionMatrix=new Ft,this.projectionMatrixInverse=new Ft,this.coordinateSystem=Ci,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,n){return super.copy(e,n),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(Yf,qf,fs),fs.x===1&&fs.y===1&&fs.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Yf,qf,fs.set(1,1,1)).invert()}updateWorldMatrix(e,n,i=!1){super.updateWorldMatrix(e,n,i),this.matrixWorld.decompose(Yf,qf,fs),fs.x===1&&fs.y===1&&fs.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Yf,qf,fs.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},Dr=new G,$A=new Oe,eM=new Oe,On=class extends Ac{constructor(e=50,n=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=n,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){let n=.5*this.getFilmHeight()/e;this.fov=rh*2*Math.atan(n),this.updateProjectionMatrix()}getFocalLength(){let e=Math.tan(Cg*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return rh*2*Math.atan(Math.tan(Cg*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,n,i){Dr.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Dr.x,Dr.y).multiplyScalar(-e/Dr.z),Dr.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(Dr.x,Dr.y).multiplyScalar(-e/Dr.z)}getViewSize(e,n){return this.getViewBounds(e,$A,eM),n.subVectors(eM,$A)}setViewOffset(e,n,i,s,r,a){this.aspect=e/n,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=this.near,n=e*Math.tan(Cg*.5*this.fov)/this.zoom,i=2*n,s=this.aspect*i,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let l=a.fullWidth,c=a.fullHeight;r+=a.offsetX*s/l,n-=a.offsetY*i/c,s*=a.width/l,i*=a.height/c}let o=this.filmOffset;o!==0&&(r+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,n,n-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.fov=this.fov,n.object.zoom=this.zoom,n.object.near=this.near,n.object.far=this.far,n.object.focus=this.focus,n.object.aspect=this.aspect,this.view!==null&&(n.object.view=Object.assign({},this.view)),n.object.filmGauge=this.filmGauge,n.object.filmOffset=this.filmOffset,n}};var Fr=class extends Ac{constructor(e=-1,n=1,i=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=n,this.top=i,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(e,n){return super.copy(e,n),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,n,i,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=n,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let e=(this.right-this.left)/(2*this.zoom),n=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=i-e,a=i+e,o=s+n,l=s-n;if(this.view!==null&&this.view.enabled){let c=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,a=r+c*this.view.width,o-=h*this.view.offsetY,l=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){let n=super.toJSON(e);return n.object.zoom=this.zoom,n.object.left=this.left,n.object.right=this.right,n.object.top=this.top,n.object.bottom=this.bottom,n.object.near=this.near,n.object.far=this.far,this.view!==null&&(n.object.view=Object.assign({},this.view)),n}};var Fo=-90,zo=1,Qo=class extends Di{constructor(e,n,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new On(Fo,zo,e,n);s.layers=this.layers,this.add(s);let r=new On(Fo,zo,e,n);r.layers=this.layers,this.add(r);let a=new On(Fo,zo,e,n);a.layers=this.layers,this.add(a);let o=new On(Fo,zo,e,n);o.layers=this.layers,this.add(o);let l=new On(Fo,zo,e,n);l.layers=this.layers,this.add(l);let c=new On(Fo,zo,e,n);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){let e=this.coordinateSystem,n=this.children.concat(),[i,s,r,a,o,l]=n;for(let c of n)this.remove(c);if(e===Ci)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===hc)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(let c of n)this.add(c),c.updateMatrixWorld()}update(e,n){this.parent===null&&this.updateMatrixWorld();let{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,l,c,h]=this.children,p=e.getRenderTarget(),u=e.getActiveCubeFace(),d=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;let A=i.texture.mipmapsAutoUpdate;i.texture.mipmapsAutoUpdate=!1;let g=!1;e.isWebGLRenderer===!0?g=e.state.buffers.depth.getReversed():g=e.reversedDepthBuffer,e.setRenderTarget(i,0,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,r),e.setRenderTarget(i,1,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,a),e.setRenderTarget(i,2,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,o),e.setRenderTarget(i,3,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,l),e.setRenderTarget(i,4,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,c),i.texture.mipmapsAutoUpdate=A,e.setRenderTarget(i,5,s),g&&e.autoClear===!1&&e.clearDepth(),e.render(n,h),e.setRenderTarget(p,u,d),e.xr.enabled=x,i.texture.needsPMREMUpdate=!0}},bh=class extends On{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e,this.perCameraFrustumCulled=!1}};var Dv="\\[\\]\\.:\\/",kw=new RegExp("["+Dv+"]","g"),Uv="[^"+Dv+"]",Ww="[^"+Dv.replace("\\.","")+"]",Xw=/((?:WC+[\/:])*)/.source.replace("WC",Uv),Yw=/(WCOD+)?/.source.replace("WCOD",Ww),qw=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Uv),Qw=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Uv),Zw=new RegExp("^"+Xw+Yw+qw+Qw+"$"),Kw=["material","materials","bones","map"],$g=class{constructor(e,n,i){let s=i||Et.parseTrackName(n);this._targetGroup=e,this._bindings=e.subscribe_(n,s)}getValue(e,n){this.bind();let i=this._targetGroup.nCachedObjects_,s=this._bindings[i];s!==void 0&&s.getValue(e,n)}setValue(e,n){let i=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=i.length;s!==r;++s)i[s].setValue(e,n)}bind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].bind()}unbind(){let e=this._bindings;for(let n=this._targetGroup.nCachedObjects_,i=e.length;n!==i;++n)e[n].unbind()}},Et=class t{constructor(e,n,i){this.path=n,this.parsedPath=i||t.parseTrackName(n),this.node=t.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,n,i){return e&&e.isAnimationObjectGroup?new t.Composite(e,n,i):new t(e,n,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(kw,"")}static parseTrackName(e){let n=Zw.exec(e);if(n===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+e);let i={nodeName:n[2],objectName:n[3],objectIndex:n[4],propertyName:n[5],propertyIndex:n[6]},s=i.nodeName&&i.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=i.nodeName.substring(s+1);Kw.indexOf(r)!==-1&&(i.nodeName=i.nodeName.substring(0,s),i.objectName=r)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,n){if(n===void 0||n===""||n==="."||n===-1||n===e.name||n===e.uuid)return e;if(e.skeleton){let i=e.skeleton.getBoneByName(n);if(i!==void 0)return i}if(e.children){let i=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===n||o.uuid===n)return o;let l=i(o.children);if(l)return l}return null},s=i(e.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,n){e[n]=this.targetObject[this.propertyName]}_getValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)e[n++]=i[s]}_getValue_arrayElement(e,n){e[n]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,n){this.resolvedProperty.toArray(e,n)}_setValue_direct(e,n){this.targetObject[this.propertyName]=e[n]}_setValue_direct_setNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++]}_setValue_array_setNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,n){let i=this.resolvedProperty;for(let s=0,r=i.length;s!==r;++s)i[s]=e[n++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,n){this.resolvedProperty[this.propertyIndex]=e[n]}_setValue_arrayElement_setNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,n){this.resolvedProperty.fromArray(e,n)}_setValue_fromArray_setNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,n){this.bind(),this.getValue(e,n)}_setValue_unbound(e,n){this.bind(),this.setValue(e,n)}bind(){let e=this.node,n=this.parsedPath,i=n.objectName,s=n.propertyName,r=n.propertyIndex;if(e||(e=t.findNode(this.rootNode,n.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){De("PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let c=n.objectIndex;switch(i){case"materials":if(!e.material){Be("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){Be("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){Be("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let h=0;h<e.length;h++)if(e[h].name===c){c=h;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){Be("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){Be("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){Be("PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(c!==void 0){if(e[c]===void 0){Be("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[c]}}let a=e[s];if(a===void 0){let c=n.nodeName;Be("PropertyBinding: Trying to update property for track: "+c+"."+s+" but it wasn't found.",e);return}let o=this.Versioning.None;this.targetObject=e,e.isMaterial===!0?o=this.Versioning.NeedsUpdate:e.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let l=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!e.geometry){Be("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){Be("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[r]!==void 0&&(r=e.morphTargetDictionary[r])}l=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(l=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(l=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[l],this.setValue=this.SetterByBindingTypeAndVersioning[l][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Et.Composite=$g;Et.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Et.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Et.prototype.GetterByBindingType=[Et.prototype._getValue_direct,Et.prototype._getValue_array,Et.prototype._getValue_arrayElement,Et.prototype._getValue_toArray];Et.prototype.SetterByBindingTypeAndVersioning=[[Et.prototype._setValue_direct,Et.prototype._setValue_direct_setNeedsUpdate,Et.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_array,Et.prototype._setValue_array_setNeedsUpdate,Et.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_arrayElement,Et.prototype._setValue_arrayElement_setNeedsUpdate,Et.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Et.prototype._setValue_fromArray,Et.prototype._setValue_fromArray_setNeedsUpdate,Et.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var KU=new Float32Array(1);var Ut=class t{constructor(e){this.value=e}clone(){return new t(this.value.clone===void 0?this.value:this.value.clone())}};var Mc=class{constructor(e=!0){this.autoStart=e,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1,De("Clock: This module has been deprecated. Please use THREE.Timer instead.")}start(){this.startTime=performance.now(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let e=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){let n=performance.now();e=(n-this.oldTime)/1e3,this.oldTime=n,this.elapsedTime+=e}return e}};var Ov=class Ov{constructor(e,n,i,s){this.elements=[1,0,0,1],e!==void 0&&this.set(e,n,i,s)}identity(){return this.set(1,0,0,1),this}fromArray(e,n=0){for(let i=0;i<4;i++)this.elements[i]=e[i+n];return this}set(e,n,i,s){let r=this.elements;return r[0]=e,r[2]=n,r[1]=i,r[3]=s,this}};Ov.prototype.isMatrix2=!0;var ev=Ov;function Iv(t,e,n,i){let s=Jw(i);switch(n){case _v:return t*e;case Av:return t*e/s.components*s.byteLength;case Nh:return t*e/s.components*s.byteLength;case zr:return t*e*2/s.components*s.byteLength;case Lh:return t*e*2/s.components*s.byteLength;case Sv:return t*e*3/s.components*s.byteLength;case Yi:return t*e*4/s.components*s.byteLength;case Ph:return t*e*4/s.components*s.byteLength;case bc:case wc:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case Cc:case Rc:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case Fh:case Hh:return Math.max(t,16)*Math.max(e,8)/4;case Oh:case zh:return Math.max(t,8)*Math.max(e,8)/2;case Gh:case Vh:case Wh:case Xh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*8;case kh:case Dc:case Yh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case qh:return Math.floor((t+3)/4)*Math.floor((e+3)/4)*16;case Qh:return Math.floor((t+4)/5)*Math.floor((e+3)/4)*16;case Zh:return Math.floor((t+4)/5)*Math.floor((e+4)/5)*16;case Kh:return Math.floor((t+5)/6)*Math.floor((e+4)/5)*16;case Jh:return Math.floor((t+5)/6)*Math.floor((e+5)/6)*16;case jh:return Math.floor((t+7)/8)*Math.floor((e+4)/5)*16;case $h:return Math.floor((t+7)/8)*Math.floor((e+5)/6)*16;case ed:return Math.floor((t+7)/8)*Math.floor((e+7)/8)*16;case td:return Math.floor((t+9)/10)*Math.floor((e+4)/5)*16;case nd:return Math.floor((t+9)/10)*Math.floor((e+5)/6)*16;case id:return Math.floor((t+9)/10)*Math.floor((e+7)/8)*16;case sd:return Math.floor((t+9)/10)*Math.floor((e+9)/10)*16;case rd:return Math.floor((t+11)/12)*Math.floor((e+9)/10)*16;case ad:return Math.floor((t+11)/12)*Math.floor((e+11)/12)*16;case od:case ld:case cd:return Math.ceil(t/4)*Math.ceil(e/4)*16;case ud:case fd:return Math.ceil(t/4)*Math.ceil(e/4)*8;case Uc:case hd:return Math.ceil(t/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${n} format.`)}function Jw(t){switch(t){case Gt:case gv:return{byteLength:1,components:1};case jo:case vv:case Xi:return{byteLength:2,components:1};case Ih:case Bh:return{byteLength:2,components:4};case Wi:case Uh:case fi:return{byteLength:4,components:1};case xv:case yv:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${t}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"187"}}));typeof window<"u"&&(window.__THREE__?De("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="187");function c1(){let t=null,e=!1,n=null,i=null;function s(r,a){i=t.requestAnimationFrame(s),n(r,a)}return{start:function(){e!==!0&&n!==null&&t!==null&&(i=t.requestAnimationFrame(s),e=!0)},stop:function(){t!==null&&t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){n=r},setContext:function(r){t=r}}}function jw(t){let e=new WeakMap;function n(o,l){let c=o.array,h=o.usage,p=c.byteLength,u=t.createBuffer();t.bindBuffer(l,u),t.bufferData(l,c,h),o.onUploadCallback();let d;if(c instanceof Float32Array)d=t.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)d=t.HALF_FLOAT;else if(c instanceof Uint16Array)o.isFloat16BufferAttribute?d=t.HALF_FLOAT:d=t.UNSIGNED_SHORT;else if(c instanceof Int16Array)d=t.SHORT;else if(c instanceof Uint32Array)d=t.UNSIGNED_INT;else if(c instanceof Int32Array)d=t.INT;else if(c instanceof Int8Array)d=t.BYTE;else if(c instanceof Uint8Array)d=t.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)d=t.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:u,type:d,bytesPerElement:c.BYTES_PER_ELEMENT,version:o.version,size:p}}function i(o,l,c){let h=l.array,p=l.updateRanges;if(t.bindBuffer(c,o),p.length===0)t.bufferSubData(c,0,h);else{p.sort((d,x)=>d.start-x.start);let u=0;for(let d=1;d<p.length;d++){let x=p[u],A=p[d];A.start<=x.start+x.count+1?x.count=Math.max(x.count,A.start+A.count-x.start):(++u,p[u]=A)}p.length=u+1;for(let d=0,x=p.length;d<x;d++){let A=p[d];t.bufferSubData(c,A.start*h.BYTES_PER_ELEMENT,h,A.start,A.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let l=e.get(o);l&&(t.deleteBuffer(l.buffer),e.delete(o))}function a(o,l){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let c=e.get(o);if(c===void 0)e.set(o,n(o,l));else if(c.version<o.version){if(c.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(c.buffer,o,l),c.version=o.version}}return{get:s,remove:r,update:a}}var $w=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,eC=`#ifdef USE_ALPHAHASH
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
#endif`,tC=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,nC=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,iC=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,sC=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,rC=`#ifdef USE_AOMAP
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
#endif`,aC=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,oC=`#ifdef USE_BATCHING
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
#endif`,lC=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,cC=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,uC=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,fC=`float G_BlinnPhong_Implicit( ) {
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
} // validated`,hC=`#ifdef USE_IRIDESCENCE
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
#endif`,dC=`#ifdef USE_BUMPMAP
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
#endif`,pC=`#if NUM_CLIPPING_PLANES > 0
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
#endif`,mC=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,gC=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,vC=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,xC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,yC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,_C=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,SC=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
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
#endif`,AC=`#define PI 3.141592653589793
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
} // validated`,MC=`vec3 transformedNormal = objectNormal;
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
#endif`,EC=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,TC=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,bC=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,wC=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,CC="gl_FragColor = linearToOutputTexel( gl_FragColor );",RC=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,DC=`#ifdef USE_ENVMAP
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
#endif`,UC=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	uniform samplerCube envMap;
	#ifdef ENVMAP_TYPE_PMREM
		float roughnessToMip( const in float roughness ) {
			float r = clamp( roughness, 0.0, 1.0 );
			return ENVMAP_MAX_LOD * r * ( 2.0 - r );
		}
	#endif
#endif`,IC=`#ifdef USE_ENVMAP
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
#endif`,BC=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,NC=`#ifdef USE_ENVMAP
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
#endif`,LC=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,PC=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,OC=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,FC=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,zC=`#ifdef USE_GRADIENTMAP
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
}`,HC=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,GC=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,VC=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,kC=`uniform bool receiveShadow;
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
#if NUM_SUN_LIGHTS > 0
	struct SunLight {
		vec3 direction;
		vec3 color;
	};
	uniform SunLight sunLights[ NUM_SUN_LIGHTS ];
	void getSunLightInfo( const in SunLight sunLight, out IncidentLight light ) {
		light.color = sunLight.color;
		light.direction = sunLight.direction;
		light.visible = true;
	}
#endif
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
#include <lightprobes_pars_fragment>`,WC=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_PMREM
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureLod( envMap, envMapRotation * worldNormal, ENVMAP_MAX_LOD );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_PMREM
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureLod( envMap, envMapRotation * reflectVec, roughnessToMip( roughness ) );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_RETROREFLECTION
		vec3 getIBLRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
			#ifdef ENVMAP_TYPE_PMREM
				vec3 retroVec = normalize( mix( viewDir, normal, pow4( roughness ) ) );
				retroVec = transformDirectionByInverseViewMatrix( retroVec, viewMatrix );
				vec4 envMapColor = textureLod( envMap, envMapRotation * retroVec, roughnessToMip( roughness ) );
				return envMapColor.rgb * envMapIntensity;
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_PMREM
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
		#ifdef USE_RETROREFLECTION
			vec3 getIBLAnisotropyRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
				#ifdef ENVMAP_TYPE_PMREM
					vec3 bentNormal = cross( bitangent, viewDir );
					bentNormal = normalize( cross( bentNormal, bitangent ) );
					bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
					return getIBLRetroRadiance( viewDir, bentNormal, roughness );
				#else
					return vec3( 0.0 );
				#endif
			}
		#endif
	#endif
#endif`,XC=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,YC=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,qC=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,QC=`varying vec3 vViewPosition;
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
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,ZC=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.045 );
material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef USE_DIFFUSE_ROUGHNESS
	float diffuseRoughnessFactor = diffuseRoughness;
	#ifdef USE_DIFFUSE_ROUGHNESSMAP
		diffuseRoughnessFactor *= texture2D( diffuseRoughnessMap, vDiffuseRoughnessMapUv ).r;
	#endif
	material.diffuseRoughness = saturate( diffuseRoughnessFactor );
#endif
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
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.045 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_RETROREFLECTION
	material.retroreflectivity = retroreflectivity;
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
#endif`,KC=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	vec2 dfg;
	vec3 multiScatteringCompensation;
	#ifdef USE_DIFFUSE_ROUGHNESS
		float diffuseRoughness;
	#endif
	#ifdef USE_RETROREFLECTION
		float retroreflectivity;
	#endif
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
		vec3 iridescenceF0Dielectric;
		vec3 iridescenceF0Metallic;
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
#ifdef USE_DIFFUSE_ROUGHNESS
	const float EON_EPSILON = 1e-7;
	float FON_DirectionalAlbedo( const in float mu, const in float roughness, const in float A ) {
		float muComp = 1.0 - mu;
		float gOverPi = muComp * ( 0.0571085289 + muComp * ( 0.491881867 + muComp * ( - 0.332181442 + muComp * 0.0714429953 ) ) );
		return A * ( 1.0 + roughness * gOverPi );
	}
	vec3 BRDF_EON( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 diffuseColor, const in float roughness ) {
		vec3 rho = saturate( diffuseColor );
		if ( roughness <= EON_EPSILON ) return BRDF_Lambert( rho );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float s = dot( lightDir, viewDir ) - dotNL * dotNV;
		float sOverT = ( s > 0.0 ) ? s / max( max( dotNL, dotNV ), EON_EPSILON ) : s;
		float A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );
		vec3 singleScatter = rho * RECIPROCAL_PI * A * ( 1.0 + roughness * sOverT );
		float averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );
		float albedoV = FON_DirectionalAlbedo( dotNV, roughness, A );
		float albedoL = FON_DirectionalAlbedo( dotNL, roughness, A );
		vec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );
		vec3 multiScatter = rhoMultiScatter * RECIPROCAL_PI
			* max( EON_EPSILON, 1.0 - albedoV )
			* max( EON_EPSILON, 1.0 - albedoL )
			/ max( EON_EPSILON, 1.0 - averageAlbedo );
		return singleScatter + multiScatter;
	}
	vec3 EON_DirectionalAlbedo( const in vec3 diffuseColor, const in float roughness, const in float dotNV ) {
		vec3 rho = saturate( diffuseColor );
		if ( roughness <= EON_EPSILON ) return rho;
		float A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );
		float directionalAlbedo = FON_DirectionalAlbedo( dotNV, roughness, A );
		float averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );
		vec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );
		return rho * directionalAlbedo + rhoMultiScatter * ( 1.0 - directionalAlbedo );
	}
#endif
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
		float roughness = max( material.clearcoatRoughness, 0.045 );
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
	float roughness = max( material.roughness, 0.045 );
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
		float alphaT = max( material.alphaT, alpha );
		float V = V_GGX_SmithCorrelated_Anisotropic( alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( alphaT, alpha, dotNH, dotTH, dotBH );
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
void computeMultiscatteringIridescence( const in vec2 fab, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec2 fab, const in vec3 specularColor, const in float specularF90, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
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
	vec3 specularBRDF = BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	#ifdef USE_RETROREFLECTION
		vec3 retroViewDir = reflect( - geometryViewDir, geometryNormal );
		vec3 retroSpecularBRDF = BRDF_GGX( directLight.direction, retroViewDir, geometryNormal, material );
		specularBRDF = mix( specularBRDF, retroSpecularBRDF, saturate( material.retroreflectivity ) );
	#endif
	reflectedLight.directSpecular += irradiance * specularBRDF * material.multiScatteringCompensation;
	vec3 halfDir = normalize( directLight.direction + geometryViewDir );
	float dotVH = saturate( dot( geometryViewDir, halfDir ) );
	vec3 F = F_Schlick( material.specularColor, material.specularF90, dotVH );
	#ifdef USE_RETROREFLECTION
		vec3 retroHalfDir = normalize( directLight.direction + retroViewDir );
		float dotRetroVH = saturate( dot( retroViewDir, retroHalfDir ) );
		vec3 retroF = F_Schlick( material.specularColor, material.specularF90, dotRetroVH );
		F = mix( F, retroF, saturate( material.retroreflectivity ) );
	#endif
	#ifdef USE_DIFFUSE_ROUGHNESS
		vec3 diffuseBRDF = BRDF_EON( directLight.direction, geometryViewDir, geometryNormal, material.diffuseColor, material.diffuseRoughness ) * ( 1.0 - material.metalness );
	#else
		vec3 diffuseBRDF = BRDF_Lambert( material.diffuseContribution );
	#endif
	reflectedLight.directDiffuse += irradiance * diffuseBRDF * ( 1.0 - F );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScattering, multiScattering );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScattering, multiScattering );
	#endif
	#ifdef USE_DIFFUSE_ROUGHNESS
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		vec3 diffuseAlbedo = EON_DirectionalAlbedo( material.diffuseColor, material.diffuseRoughness, dotNV ) * ( 1.0 - material.metalness );
		vec3 diffuse = irradiance * RECIPROCAL_PI * diffuseAlbedo * ( 1.0 - singleScattering - multiScattering );
	#else
		vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - singleScattering - multiScattering );
	#endif
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		sheenSpecularIndirect += irradiance * material.sheenColor * sheenAlbedo * RECIPROCAL_PI;
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
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( material.dfg, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceF0Metallic, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( material.dfg, material.diffuseColor, material.specularF90, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	#ifdef USE_DIFFUSE_ROUGHNESS
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		vec3 diffuse = EON_DirectionalAlbedo( material.diffuseColor, material.diffuseRoughness, dotNV ) * ( 1.0 - material.metalness ) * ( 1.0 - totalScatteringDielectric );
	#else
		vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	#endif
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
}`,JC=`
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
		vec3 iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		vec3 iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( iridescenceFresnelDielectric, iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0Dielectric = Schlick_to_F0( iridescenceFresnelDielectric, 1.0, dotNVi );
		material.iridescenceF0Metallic = Schlick_to_F0( iridescenceFresnelMetallic, 1.0, dotNVi );
	}
#endif
#ifdef STANDARD
	float dotNVms = saturate( dot( geometryNormal, geometryViewDir ) );
	material.dfg = texture2D( dfgLUT, vec2( material.roughness, dotNVms ) ).rg;
	#if ( NUM_SUN_LIGHTS > 0 || NUM_DIR_LIGHTS > 0 || NUM_POINT_LIGHTS > 0 || NUM_SPOT_LIGHTS > 0 )
		float EssMs = material.dfg.x + material.dfg.y;
		material.multiScatteringCompensation = 1.0 + material.specularColorBlended * ( 1.0 / EssMs - 1.0 );
	#endif
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
#if ( NUM_SUN_LIGHTS > 0 ) && defined( RE_Direct )
	SunLight sunLight;
	#if defined( USE_SHADOWMAP ) && NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHTS; i ++ ) {
		sunLight = sunLights[ i ];
		getSunLightInfo( sunLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SUN_LIGHT_SHADOWS )
		sunLightShadow = sunLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getSunShadow( sunShadowMap[ i ], sunLightShadow, UNROLLED_LOOP_INDEX ) : 1.0;
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
	#if NUM_LIGHT_PROBE_GRIDS > 0
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		int probeGrid = - 1;
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {
			if ( isInsideLightProbeGrid( probeWorldPos, probeWorldNormal, probesMin[ i ], probesMax[ i ], probesResolution[ i ] ) ) probeGrid = UNROLLED_LOOP_INDEX;
		}
		#pragma unroll_loop_end
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {
			if ( probeGrid == UNROLLED_LOOP_INDEX ) irradiance += getLightProbeGridIrradiance( probesSH[ i ], probesMin[ i ], probesMax[ i ], probesResolution[ i ], probeWorldPos, probeWorldNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,jC=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_PMREM )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		vec3 iblRadiance = getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		vec3 iblRadiance = getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_RETROREFLECTION
		#ifdef USE_ANISOTROPY
			vec3 retroIBLRadiance = getIBLAnisotropyRetroRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
		#else
			vec3 retroIBLRadiance = getIBLRetroRadiance( geometryViewDir, geometryNormal, material.roughness );
		#endif
		iblRadiance = mix( iblRadiance, retroIBLRadiance, saturate( material.retroreflectivity ) );
	#endif
	radiance += iblRadiance;
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,$C=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,e2=`#if NUM_LIGHT_PROBE_GRIDS > 0
uniform highp sampler3D probesSH[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesMin[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesMax[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesResolution[ NUM_LIGHT_PROBE_GRIDS ];
bool isInsideLightProbeGrid( vec3 worldPos, vec3 worldNormal, vec3 probesMin, vec3 probesMax, vec3 probesResolution ) {
	vec3 spacing = ( probesMax - probesMin ) / probesResolution;
	vec3 samplePos = worldPos + worldNormal * spacing * 0.5;
	vec3 margin = spacing * 0.01;
	return all( greaterThanEqual( samplePos, probesMin - margin ) ) && all( lessThanEqual( samplePos, probesMax + margin ) );
}
vec3 getLightProbeGridIrradiance( highp sampler3D probesSH, vec3 probesMin, vec3 probesMax, vec3 probesResolution, vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 probeSpacing = gridRange / res;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.5 / res, 1.0 - 0.5 / res );
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
#endif`,t2=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,n2=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,i2=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,s2=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,r2=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,a2=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,o2=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
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
#endif`,l2=`#if defined( USE_POINTS_UV )
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
#endif`,c2=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,u2=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,f2=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,h2=`#if defined( USE_MORPHCOLORS )
	#if defined( USE_COLOR_ALPHA )
		vColor *= morphTargetBaseInfluence;
	#elif defined( USE_COLOR )
		vColor.rgb *= morphTargetBaseInfluence;
	#endif
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor.rgb += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,d2=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,p2=`#ifdef USE_MORPHTARGETS
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
#endif`,m2=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,g2=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
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
vec3 nonPerturbedNormal = normal;`,v2=`#ifdef USE_NORMALMAP_OBJECTSPACE
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
#endif`,x2=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,y2=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,_2=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,S2=`#ifdef USE_NORMALMAP
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
#endif`,A2=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,M2=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,E2=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,T2=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,b2=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,w2=`vec3 packNormalToRGB( const in vec3 normal ) {
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
}`,C2=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,R2=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,D2=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,U2=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,I2=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,B2=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,N2=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		#define SUN_LIGHT_CASCADES 2
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#else
			uniform sampler2D sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#endif
		uniform mat4 sunShadowMatrix[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		uniform vec4 sunShadowCascade[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
		struct SunLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SunLightShadow sunLightShadows[ NUM_SUN_LIGHT_SHADOWS ];
	#endif
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
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				vec2 radius = shadowRadius * texelSize;
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
	#if NUM_SUN_LIGHT_SHADOWS > 0
		float getSunShadow(
			#if defined( SHADOWMAP_TYPE_PCF )
				sampler2DShadow shadowMap,
			#else
				sampler2D shadowMap,
			#endif
			SunLightShadow sunLightShadow,
			int shadowIndex
		) {
			vec4 shadowWorldPosition = vec4( vSunShadowWorldPosition.xyz + vSunShadowWorldNormal * sunLightShadow.shadowNormalBias, 1.0 );
			float viewDepth = vSunShadowWorldPosition.w;
			int cascadeOffset = shadowIndex * SUN_LIGHT_CASCADES;
			float shadow = 1.0;
			for ( int i = SUN_LIGHT_CASCADES - 1; i >= 0; i -- ) {
				vec4 cascade = sunShadowCascade[ cascadeOffset + i ];
				if ( viewDepth >= cascade.x && viewDepth < cascade.y ) {
					float cascadeShadow = getShadow(
						shadowMap,
						sunLightShadow.shadowMapSize,
						sunLightShadow.shadowIntensity,
						sunLightShadow.shadowBias,
						sunLightShadow.shadowRadius,
						sunShadowMatrix[ cascadeOffset + i ] * shadowWorldPosition
					);
					shadow = mix( cascadeShadow, shadow, smoothstep( cascade.z, cascade.y, viewDepth ) );
				}
			}
			return shadow;
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
			float lightDistance = length( lightToPosition );
			float depthRange = shadowCameraFar - shadowCameraNear;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float depthA = - shadowCameraNear / depthRange - shadowBias;
				float depthB = shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );
			#else
				float depthA = shadowCameraFar / depthRange + shadowBias;
				float depthB = - shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			shadow = 0.0;
			for ( int i = 0; i < 5; i ++ ) {
				vec2 offset = vogelDiskSample( i, 5, phi );
				vec3 direction = bd3D + ( tangent * offset.x + bitangent * offset.y ) * texelSize;
				vec3 absDirection = abs( direction );
				float dp = depthA + depthB / max( max( absDirection.x, absDirection.y ), absDirection.z );
				shadow += texture( shadowMap, vec4( direction, dp ) );
			}
			shadow *= 0.2;
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
#endif`,L2=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
	#endif
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
#endif`,P2=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_SUN_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_SUN_LIGHT_SHADOWS > 0
		vSunShadowWorldPosition = vec4( worldPosition.xyz, - mvPosition.z );
		vSunShadowWorldNormal = shadowWorldNormal;
	#endif
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
#endif`,O2=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHT_SHADOWS; i ++ ) {
		sunLight = sunLightShadows[ i ];
		shadow *= receiveShadow ? getSunShadow( sunShadowMap[ i ], sunLight, UNROLLED_LOOP_INDEX ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
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
}`,F2=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,z2=`#ifdef USE_SKINNING
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
#endif`,H2=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,G2=`#ifdef USE_SKINNING
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
#endif`,V2=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,k2=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,W2=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,X2=`#ifndef saturate
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
vec3 CustomToneMapping( vec3 color ) { return color; }`,Y2=`#ifdef USE_TRANSMISSION
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
#endif`,q2=`#ifdef USE_TRANSMISSION
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
#endif`,Q2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	varying vec2 vDiffuseRoughnessMapUv;
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
#endif`,Z2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	uniform mat3 diffuseRoughnessMapTransform;
	varying vec2 vDiffuseRoughnessMapUv;
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
#endif`,K2=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
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
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	vDiffuseRoughnessMapUv = ( diffuseRoughnessMapTransform * vec3( DIFFUSE_ROUGHNESSMAP_UV, 1 ) ).xy;
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
#endif`,J2=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,j2=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,$2=`uniform sampler2D t2D;
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
}`,eR=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,tR=`uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <envmap_common_pars_fragment>
void main() {
	#ifdef ENVMAP_TYPE_PMREM
		vec4 texColor = textureLod( envMap, backgroundRotation * vWorldDirection, roughnessToMip( backgroundBlurriness ) );
	#else
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,nR=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,iR=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,sR=`#include <common>
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
}`,rR=`#if DEPTH_PACKING == 3200
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
}`,aR=`#define DISTANCE
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
}`,oR=`#define DISTANCE
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
}`,lR=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,cR=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,uR=`uniform float scale;
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
}`,fR=`uniform vec3 diffuse;
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
}`,hR=`#include <common>
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
}`,dR=`uniform vec3 diffuse;
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
}`,pR=`#define LAMBERT
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
}`,mR=`#define LAMBERT
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
}`,gR=`#define MATCAP
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
}`,vR=`#define MATCAP
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
}`,xR=`#define NORMAL
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
}`,yR=`#define NORMAL
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
}`,_R=`#define PHONG
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
}`,SR=`#define PHONG
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
}`,AR=`#define STANDARD
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
}`,MR=`#define STANDARD
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
#ifdef USE_DIFFUSE_ROUGHNESS
	uniform float diffuseRoughness;
	#ifdef USE_DIFFUSE_ROUGHNESSMAP
		uniform sampler2D diffuseRoughnessMap;
	#endif
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_RETROREFLECTION
	uniform float retroreflectivity;
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
}`,ER=`#define TOON
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
}`,TR=`#define TOON
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
}`,bR=`uniform float size;
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
}`,wR=`uniform vec3 diffuse;
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
}`,CR=`#include <common>
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
}`,RR=`uniform vec3 color;
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
}`,DR=`uniform float rotation;
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
}`,UR=`uniform vec3 diffuse;
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
}`,Ve={alphahash_fragment:$w,alphahash_pars_fragment:eC,alphamap_fragment:tC,alphamap_pars_fragment:nC,alphatest_fragment:iC,alphatest_pars_fragment:sC,aomap_fragment:rC,aomap_pars_fragment:aC,batching_pars_vertex:oC,batching_vertex:lC,begin_vertex:cC,beginnormal_vertex:uC,bsdfs:fC,iridescence_fragment:hC,bumpmap_pars_fragment:dC,clipping_planes_fragment:pC,clipping_planes_pars_fragment:mC,clipping_planes_pars_vertex:gC,clipping_planes_vertex:vC,color_fragment:xC,color_pars_fragment:yC,color_pars_vertex:_C,color_vertex:SC,common:AC,defaultnormal_vertex:MC,displacementmap_pars_vertex:EC,displacementmap_vertex:TC,emissivemap_fragment:bC,emissivemap_pars_fragment:wC,colorspace_fragment:CC,colorspace_pars_fragment:RC,envmap_fragment:DC,envmap_common_pars_fragment:UC,envmap_pars_fragment:IC,envmap_pars_vertex:BC,envmap_physical_pars_fragment:WC,envmap_vertex:NC,fog_vertex:LC,fog_pars_vertex:PC,fog_fragment:OC,fog_pars_fragment:FC,gradientmap_pars_fragment:zC,lightmap_pars_fragment:HC,lights_lambert_fragment:GC,lights_lambert_pars_fragment:VC,lights_pars_begin:kC,lights_toon_fragment:XC,lights_toon_pars_fragment:YC,lights_phong_fragment:qC,lights_phong_pars_fragment:QC,lights_physical_fragment:ZC,lights_physical_pars_fragment:KC,lights_fragment_begin:JC,lights_fragment_maps:jC,lights_fragment_end:$C,lightprobes_pars_fragment:e2,logdepthbuf_fragment:t2,logdepthbuf_pars_fragment:n2,logdepthbuf_pars_vertex:i2,logdepthbuf_vertex:s2,map_fragment:r2,map_pars_fragment:a2,map_particle_fragment:o2,map_particle_pars_fragment:l2,metalnessmap_fragment:c2,metalnessmap_pars_fragment:u2,morphinstance_vertex:f2,morphcolor_vertex:h2,morphnormal_vertex:d2,morphtarget_pars_vertex:p2,morphtarget_vertex:m2,normal_fragment_begin:g2,normal_fragment_maps:v2,normal_pars_fragment:x2,normal_pars_vertex:y2,normal_vertex:_2,normalmap_pars_fragment:S2,clearcoat_normal_fragment_begin:A2,clearcoat_normal_fragment_maps:M2,clearcoat_pars_fragment:E2,iridescence_pars_fragment:T2,opaque_fragment:b2,packing:w2,premultiplied_alpha_fragment:C2,project_vertex:R2,dithering_fragment:D2,dithering_pars_fragment:U2,roughnessmap_fragment:I2,roughnessmap_pars_fragment:B2,shadowmap_pars_fragment:N2,shadowmap_pars_vertex:L2,shadowmap_vertex:P2,shadowmask_pars_fragment:O2,skinbase_vertex:F2,skinning_pars_vertex:z2,skinning_vertex:H2,skinnormal_vertex:G2,specularmap_fragment:V2,specularmap_pars_fragment:k2,tonemapping_fragment:W2,tonemapping_pars_fragment:X2,transmission_fragment:Y2,transmission_pars_fragment:q2,uv_pars_fragment:Q2,uv_pars_vertex:Z2,uv_vertex:K2,worldpos_vertex:J2,background_vert:j2,background_frag:$2,backgroundCube_vert:eR,backgroundCube_frag:tR,cube_vert:nR,cube_frag:iR,depth_vert:sR,depth_frag:rR,distance_vert:aR,distance_frag:oR,equirect_vert:lR,equirect_frag:cR,linedashed_vert:uR,linedashed_frag:fR,meshbasic_vert:hR,meshbasic_frag:dR,meshlambert_vert:pR,meshlambert_frag:mR,meshmatcap_vert:gR,meshmatcap_frag:vR,meshnormal_vert:xR,meshnormal_frag:yR,meshphong_vert:_R,meshphong_frag:SR,meshphysical_vert:AR,meshphysical_frag:MR,meshtoon_vert:ER,meshtoon_frag:TR,points_vert:bR,points_frag:wR,shadow_vert:CR,shadow_frag:RR,sprite_vert:DR,sprite_frag:UR},ge={common:{diffuse:{value:new Ye(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Ne},alphaMap:{value:null},alphaMapTransform:{value:new Ne},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Ne}},envmap:{envMap:{value:null},envMapRotation:{value:new Ne},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Ne}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Ne}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Ne},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Ne},normalScale:{value:new Oe(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Ne},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Ne}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Ne}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Ne}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Ye(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},sunLights:{value:[],properties:{direction:{},color:{}}},sunLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},sunShadowMatrix:{value:[]},sunShadowCascade:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:[]},probesMin:{value:[]},probesMax:{value:[]},probesResolution:{value:[]}},points:{diffuse:{value:new Ye(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Ne},alphaTest:{value:0},uvTransform:{value:new Ne}},sprite:{diffuse:{value:new Ye(16777215)},opacity:{value:1},center:{value:new Oe(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Ne},alphaMap:{value:null},alphaMapTransform:{value:new Ne},alphaTest:{value:0}}},ys={basic:{uniforms:Rn([ge.common,ge.specularmap,ge.envmap,ge.aomap,ge.lightmap,ge.fog]),vertexShader:Ve.meshbasic_vert,fragmentShader:Ve.meshbasic_frag},lambert:{uniforms:Rn([ge.common,ge.specularmap,ge.envmap,ge.aomap,ge.lightmap,ge.emissivemap,ge.bumpmap,ge.normalmap,ge.displacementmap,ge.fog,ge.lights,{emissive:{value:new Ye(0)},envMapIntensity:{value:1}}]),vertexShader:Ve.meshlambert_vert,fragmentShader:Ve.meshlambert_frag},phong:{uniforms:Rn([ge.common,ge.specularmap,ge.envmap,ge.aomap,ge.lightmap,ge.emissivemap,ge.bumpmap,ge.normalmap,ge.displacementmap,ge.fog,ge.lights,{emissive:{value:new Ye(0)},specular:{value:new Ye(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:Ve.meshphong_vert,fragmentShader:Ve.meshphong_frag},standard:{uniforms:Rn([ge.common,ge.envmap,ge.aomap,ge.lightmap,ge.emissivemap,ge.bumpmap,ge.normalmap,ge.displacementmap,ge.roughnessmap,ge.metalnessmap,ge.fog,ge.lights,{emissive:{value:new Ye(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Ve.meshphysical_vert,fragmentShader:Ve.meshphysical_frag},toon:{uniforms:Rn([ge.common,ge.aomap,ge.lightmap,ge.emissivemap,ge.bumpmap,ge.normalmap,ge.displacementmap,ge.gradientmap,ge.fog,ge.lights,{emissive:{value:new Ye(0)}}]),vertexShader:Ve.meshtoon_vert,fragmentShader:Ve.meshtoon_frag},matcap:{uniforms:Rn([ge.common,ge.bumpmap,ge.normalmap,ge.displacementmap,ge.fog,{matcap:{value:null}}]),vertexShader:Ve.meshmatcap_vert,fragmentShader:Ve.meshmatcap_frag},points:{uniforms:Rn([ge.points,ge.fog]),vertexShader:Ve.points_vert,fragmentShader:Ve.points_frag},dashed:{uniforms:Rn([ge.common,ge.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Ve.linedashed_vert,fragmentShader:Ve.linedashed_frag},depth:{uniforms:Rn([ge.common,ge.displacementmap]),vertexShader:Ve.depth_vert,fragmentShader:Ve.depth_frag},normal:{uniforms:Rn([ge.common,ge.bumpmap,ge.normalmap,ge.displacementmap,{opacity:{value:1}}]),vertexShader:Ve.meshnormal_vert,fragmentShader:Ve.meshnormal_frag},sprite:{uniforms:Rn([ge.sprite,ge.fog]),vertexShader:Ve.sprite_vert,fragmentShader:Ve.sprite_frag},background:{uniforms:{uvTransform:{value:new Ne},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Ve.background_vert,fragmentShader:Ve.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Ne}},vertexShader:Ve.backgroundCube_vert,fragmentShader:Ve.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Ve.cube_vert,fragmentShader:Ve.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Ve.equirect_vert,fragmentShader:Ve.equirect_frag},distance:{uniforms:Rn([ge.common,ge.displacementmap,{referencePosition:{value:new G},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Ve.distance_vert,fragmentShader:Ve.distance_frag},shadow:{uniforms:Rn([ge.lights,ge.fog,{color:{value:new Ye(0)},opacity:{value:1}}]),vertexShader:Ve.shadow_vert,fragmentShader:Ve.shadow_frag}};ys.physical={uniforms:Rn([ys.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Ne},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Ne},clearcoatNormalScale:{value:new Oe(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Ne},diffuseRoughness:{value:0},diffuseRoughnessMap:{value:null},diffuseRoughnessMapTransform:{value:new Ne},dispersion:{value:0},retroreflectivity:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Ne},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Ne},sheen:{value:0},sheenColor:{value:new Ye(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Ne},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Ne},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Ne},transmissionSamplerSize:{value:new Oe},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Ne},attenuationDistance:{value:0},attenuationColor:{value:new Ye(0)},specularColor:{value:new Ye(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Ne},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Ne},anisotropyVector:{value:new Oe},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Ne}}]),vertexShader:Ve.meshphysical_vert,fragmentShader:Ve.meshphysical_frag};var dd={r:0,b:0,g:0},IR=new Ft,u1=new Ne;u1.set(-1,0,0,0,1,0,0,0,1);function BR(t,e,n,i,s,r){let a=new Ye(0),o=s===!0?0:1,l,c,h=null,p=0,u=null;function d(m){let y=m.isScene===!0?m.background:null;if(y&&y.isTexture){let _=m.backgroundBlurriness>0;y=e.get(y,_)}return y}function x(m){let y=!1,_=d(m);_===null?g(a,o):_&&_.isColor&&(g(_,1),y=!0);let T=t.xr.getEnvironmentBlendMode();T==="additive"?n.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,r),(t.autoClear||y)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil))}function A(m,y){let _=d(y);_&&_.isCubeTexture?(c===void 0&&(c=new Qn(new qo(1,1,1),new hn({name:"BackgroundCubeMaterial",uniforms:Ta(ys.backgroundCube.uniforms),vertexShader:ys.backgroundCube.vertexShader,fragmentShader:ys.backgroundCube.fragmentShader,side:Qt,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),c.geometry.deleteAttribute("uv"),c.onBeforeRender=function(T,b,v){this.matrixWorld.copyPosition(v.matrixWorld)},Object.defineProperty(c.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(c)),c.material.uniforms.envMap.value=_,c.material.uniforms.backgroundBlurriness.value=y.backgroundBlurriness,c.material.uniforms.backgroundIntensity.value=y.backgroundIntensity,c.material.uniforms.backgroundRotation.value.setFromMatrix4(IR.makeRotationFromEuler(y.backgroundRotation)).transpose(),_.isRenderTargetTexture===!1&&c.material.uniforms.backgroundRotation.value.premultiply(u1),c.material.toneMapped=je.getTransfer(_.colorSpace)!==ct,(h!==_||p!==_.version||u!==t.toneMapping)&&(c.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),c.layers.enableAll(),m.unshift(c,c.geometry,c.material,0,0,null)):_&&_.isTexture&&(l===void 0&&(l=new Qn(new Aa(2,2),new hn({name:"BackgroundMaterial",uniforms:Ta(ys.background.uniforms),vertexShader:ys.background.vertexShader,fragmentShader:ys.background.fragmentShader,side:ms,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(l)),l.material.uniforms.t2D.value=_,l.material.uniforms.backgroundIntensity.value=y.backgroundIntensity,l.material.toneMapped=je.getTransfer(_.colorSpace)!==ct,_.matrixAutoUpdate===!0&&_.updateMatrix(),l.material.uniforms.uvTransform.value.copy(_.matrix),(h!==_||p!==_.version||u!==t.toneMapping)&&(l.material.needsUpdate=!0,h=_,p=_.version,u=t.toneMapping),l.layers.enableAll(),m.unshift(l,l.geometry,l.material,0,0,null))}function g(m,y){m.getRGB(dd,Rv(t)),n.buffers.color.setClear(dd.r,dd.g,dd.b,y,r)}function f(){c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0),l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0)}return{getClearColor:function(){return a},setClearColor:function(m,y=1){a.set(m),o=y,g(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(m){o=m,g(a,o)},render:x,addToRenderList:A,dispose:f}}function NR(t,e){let n=t.getParameter(t.MAX_VERTEX_ATTRIBS),i={},s=u(null),r=s,a=!1;function o(B,F,X,D,V){let q=!1,H=p(B,D,X,F);r!==H&&(r=H,c(r.object)),q=d(B,D,X,V),q&&x(B,D,X,V),V!==null&&e.update(V,t.ELEMENT_ARRAY_BUFFER),(q||a)&&(a=!1,_(B,F,X,D),V!==null&&t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,e.get(V).buffer))}function l(){return t.createVertexArray()}function c(B){return t.bindVertexArray(B)}function h(B){return t.deleteVertexArray(B)}function p(B,F,X,D){let V=D.wireframe===!0,q=i[F.id];q===void 0&&(q={},i[F.id]=q);let H=B.isInstancedMesh===!0?B.id:0,te=q[H];te===void 0&&(te={},q[H]=te);let k=te[X.id];k===void 0&&(k={},te[X.id]=k);let $=k[V];return $===void 0&&($=u(l()),k[V]=$),$}function u(B){let F=[],X=[],D=[];for(let V=0;V<n;V++)F[V]=0,X[V]=0,D[V]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:F,enabledAttributes:X,attributeDivisors:D,object:B,attributes:{},index:null}}function d(B,F,X,D){let V=r.attributes,q=F.attributes,H=0,te=X.getAttributes();for(let k in te)if(te[k].location>=0){let ee=V[k],_e=q[k];if(_e===void 0&&(k==="instanceMatrix"&&B.instanceMatrix&&(_e=B.instanceMatrix),k==="instanceColor"&&B.instanceColor&&(_e=B.instanceColor)),ee===void 0||ee.attribute!==_e||_e&&ee.data!==_e.data)return!0;H++}return r.attributesNum!==H||r.index!==D}function x(B,F,X,D){let V={},q=F.attributes,H=0,te=X.getAttributes();for(let k in te)if(te[k].location>=0){let ee=q[k];ee===void 0&&(k==="instanceMatrix"&&B.instanceMatrix&&(ee=B.instanceMatrix),k==="instanceColor"&&B.instanceColor&&(ee=B.instanceColor));let _e={};_e.attribute=ee,ee&&ee.data&&(_e.data=ee.data),V[k]=_e,H++}r.attributes=V,r.attributesNum=H,r.index=D}function A(){let B=r.newAttributes;for(let F=0,X=B.length;F<X;F++)B[F]=0}function g(B){f(B,0)}function f(B,F){let X=r.newAttributes,D=r.enabledAttributes,V=r.attributeDivisors;X[B]=1,D[B]===0&&(t.enableVertexAttribArray(B),D[B]=1),V[B]!==F&&(t.vertexAttribDivisor(B,F),V[B]=F)}function m(){let B=r.newAttributes,F=r.enabledAttributes;for(let X=0,D=F.length;X<D;X++)F[X]!==B[X]&&(t.disableVertexAttribArray(X),F[X]=0)}function y(B,F,X,D,V,q,H){H===!0?t.vertexAttribIPointer(B,F,X,V,q):t.vertexAttribPointer(B,F,X,D,V,q)}function _(B,F,X,D){A();let V=D.attributes,q=X.getAttributes(),H=F.defaultAttributeValues;for(let te in q){let k=q[te];if(k.location>=0){let $=V[te];if($===void 0&&(te==="instanceMatrix"&&B.instanceMatrix&&($=B.instanceMatrix),te==="instanceColor"&&B.instanceColor&&($=B.instanceColor)),$!==void 0){let ee=$.normalized,_e=$.itemSize,Ie=e.get($);if(Ie===void 0)continue;let Q=Ie.buffer,de=Ie.type,xe=Ie.bytesPerElement,Te=de===t.INT||de===t.UNSIGNED_INT||$.gpuType===Uh;if($.isInterleavedBufferAttribute){let pe=$.data,be=pe.isInterleavedBuffer===!0?pe.stride:pe.itemSize,qe=$.offset;if(pe.isInstancedInterleavedBuffer||pe.isInstancedBufferAttribute){for(let Ce=0;Ce<k.locationSize;Ce++)f(k.location+Ce,pe.meshPerAttribute);B.isInstancedMesh!==!0&&D._maxInstanceCount===void 0&&(D._maxInstanceCount=pe.meshPerAttribute*pe.count)}else for(let Ce=0;Ce<k.locationSize;Ce++)g(k.location+Ce);t.bindBuffer(t.ARRAY_BUFFER,Q);for(let Ce=0;Ce<k.locationSize;Ce++)y(k.location+Ce,_e/k.locationSize,de,ee,be*xe,(qe+_e/k.locationSize*Ce)*xe,Te)}else{if($.isInstancedBufferAttribute){for(let pe=0;pe<k.locationSize;pe++)f(k.location+pe,$.meshPerAttribute);B.isInstancedMesh!==!0&&D._maxInstanceCount===void 0&&(D._maxInstanceCount=$.meshPerAttribute*$.count)}else for(let pe=0;pe<k.locationSize;pe++)g(k.location+pe);t.bindBuffer(t.ARRAY_BUFFER,Q);for(let pe=0;pe<k.locationSize;pe++)y(k.location+pe,_e/k.locationSize,de,ee,_e*xe,_e/k.locationSize*pe*xe,Te)}}else if(H!==void 0){let ee=H[te];if(ee!==void 0)switch(ee.length){case 2:t.vertexAttrib2fv(k.location,ee);break;case 3:t.vertexAttrib3fv(k.location,ee);break;case 4:t.vertexAttrib4fv(k.location,ee);break;default:t.vertexAttrib1fv(k.location,ee)}}}}m()}function T(){N();for(let B in i){let F=i[B];for(let X in F){let D=F[X];for(let V in D){let q=D[V];for(let H in q)h(q[H].object),delete q[H];delete D[V]}}delete i[B]}}function b(B){if(i[B.id]===void 0)return;let F=i[B.id];for(let X in F){let D=F[X];for(let V in D){let q=D[V];for(let H in q)h(q[H].object),delete q[H];delete D[V]}}delete i[B.id]}function v(B){for(let F in i){let X=i[F];for(let D in X){let V=X[D];if(V[B.id]===void 0)continue;let q=V[B.id];for(let H in q)h(q[H].object),delete q[H];delete V[B.id]}}}function E(B){for(let F in i){let X=i[F],D=B.isInstancedMesh===!0?B.id:0,V=X[D];if(V!==void 0){for(let q in V){let H=V[q];for(let te in H)h(H[te].object),delete H[te];delete V[q]}delete X[D],Object.keys(X).length===0&&delete i[F]}}}function N(){C(),a=!0,r!==s&&(r=s,c(r.object))}function C(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:N,resetDefaultState:C,dispose:T,releaseStatesOfGeometry:b,releaseStatesOfObject:E,releaseStatesOfProgram:v,initAttributes:A,enableAttribute:g,disableUnusedAttributes:m}}function LR(t,e,n){let i;function s(l){i=l}function r(l,c){t.drawArrays(i,l,c),n.update(c,i,1)}function a(l,c,h){h!==0&&(t.drawArraysInstanced(i,l,c,h),n.update(c,i,h))}function o(l,c,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,l,0,c,0,h);let u=0;for(let d=0;d<h;d++)u+=c[d];n.update(u,i,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function PR(t,e,n,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){let v=e.get("EXT_texture_filter_anisotropic");s=t.getParameter(v.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(v){return!(v!==Yi&&i.convert(v)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(v){let E=v===Xi&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(v!==Gt&&v!==fi&&!E&&i.convert(v)!==t.getParameter(t.IMPLEMENTATION_COLOR_READ_TYPE))}function l(v){if(v==="highp"){if(t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.HIGH_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.HIGH_FLOAT).precision>0)return"highp";v="mediump"}return v==="mediump"&&t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.MEDIUM_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=n.precision!==void 0?n.precision:"highp",h=l(c);h!==c&&(De("WebGLRenderer:",c,"not supported, using",h,"instead."),c=h);let p=n.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),u=n.logarithmicDepthBuffer===!0&&p===!1;n.reversedDepthBuffer===!0&&p===!1&&De("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer."),n.logarithmicDepthBuffer===!0&&p===!0&&De("WebGLRenderer: Logarithmic depth buffer is ignored when reversed depth buffer is enabled.");let d=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS),x=t.getParameter(t.MAX_VERTEX_TEXTURE_IMAGE_UNITS),A=t.getParameter(t.MAX_TEXTURE_SIZE),g=t.getParameter(t.MAX_CUBE_MAP_TEXTURE_SIZE),f=t.getParameter(t.MAX_VERTEX_ATTRIBS),m=t.getParameter(t.MAX_VERTEX_UNIFORM_VECTORS),y=t.getParameter(t.MAX_VARYING_VECTORS),_=t.getParameter(t.MAX_FRAGMENT_UNIFORM_VECTORS),T=t.getParameter(t.MAX_SAMPLES),b=t.getParameter(t.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:a,textureTypeReadable:o,precision:c,logarithmicDepthBuffer:u,reversedDepthBuffer:p,maxTextures:d,maxVertexTextures:x,maxTextureSize:A,maxCubemapSize:g,maxAttributes:f,maxVertexUniforms:m,maxVaryings:y,maxFragmentUniforms:_,maxSamples:T,samples:b}}function OR(t){let e=this,n=null,i=0,s=!1,r=!1,a=new Gi,o=new Ne,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(p,u){let d=p.length!==0||u||i!==0||s;return s=u,i=p.length,d},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(p,u){n=h(p,u,0)},this.setState=function(p,u,d){let x=p.clippingPlanes,A=p.clipIntersection,g=p.clipShadows,f=t.get(p);if(!s||x===null||x.length===0||r&&!g)r?h(null):c();else{let m=r?0:i,y=m*4,_=f.clippingState||null;l.value=_,_=h(x,u,y,d);for(let T=0;T!==y;++T)_[T]=n[T];f.clippingState=_,this.numIntersection=A?this.numPlanes:0,this.numPlanes+=m}};function c(){l.value!==n&&(l.value=n,l.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(p,u,d,x){let A=p!==null?p.length:0,g=null;if(A!==0){if(g=l.value,x!==!0||g===null){let f=d+A*4,m=u.matrixWorldInverse;o.getNormalMatrix(m),(g===null||g.length<f)&&(g=new Float32Array(f));for(let y=0,_=d;y!==A;++y,_+=4)a.copy(p[y]).applyMatrix4(m,o),a.normal.toArray(g,_),g[_+3]=a.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=A,e.numIntersection=0,g}}var Pc=class extends zt{constructor(e=1,n={}){super(e,e,n),this.isWebGLCubeRenderTarget=!0;let i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new xc(s),this._setTextureOptions(n),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,n){this.texture.type=n.type,this.texture.colorSpace=n.colorSpace,this.texture.generateMipmaps=n.generateMipmaps,this.texture.minFilter=n.minFilter,this.texture.magFilter=n.magFilter;let i={uniforms:{tEquirect:{value:null}},vertexShader:`

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
			`},s=new qo(5,5,5),r=new hn({name:"CubemapFromEquirect",uniforms:Ta(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:Qt,blending:ki});r.uniforms.tEquirect.value=n;let a=new Qn(s,r),o=n.minFilter;return n.minFilter===Xs&&(n.minFilter=Dt),new Qo(1,10,this).update(e,a),n.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,n=!0,i=!0,s=!0){let r=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(n,i,s);e.setRenderTarget(r)}},VM=256,FR=3,zR=20,f1=256,h1=16,HR=3,GR=new G,VR=new Ye,md=class t{constructor(e){this._renderer=e,this._cubeSize=0,this._sourceTarget=null,this._cubeCamera=new Qo(1,10,null),this._boxMesh=new Qn(new qo(5,5,5),null),this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null,this._integrationMaterial=null}fromScene(e,n=0,i=.1,s=100,r={}){let{size:a=256,position:o=GR}=r,l=this._renderer;this._setSize(a);let c=this._allocateTarget(),h=this._getSourceTarget(!0);n>0&&(h.texture.mipmapsAutoUpdate=!1);let p=l.autoClear,u=l.autoClearColor,d=l.autoClearDepth,x=l.autoClearStencil,A=e.background;l.autoClear=!0,l.autoClearColor=!0,l.autoClearDepth=!0,l.autoClearStencil=!0,A===null&&(e.background=l.getClearColor(VR));let g=new Qo(i,s,h);return g.position.copy(o),g.update(l,e),l.autoClear=p,l.autoClearColor=u,l.autoClearDepth=d,l.autoClearStencil=x,e.background=A,n>0&&(h.texture.mipmapsAutoUpdate=!0,this._blur(c,n)),this._applyPMREM(c),c}fromEquirectangular(e,n=null){return this._fromTexture(e,n)}fromCubemap(e,n=null){return this._fromTexture(e,n)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=XM(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=WM(),this._compileMaterial(this._equirectMaterial))}dispose(){this._sourceTarget!==null&&this._sourceTarget.dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._integrationMaterial!==null&&this._integrationMaterial.dispose(),this._boxMesh.geometry.dispose()}static lodToRoughness(e,n){return n>0?1-Math.sqrt(1-e/n):0}_setSize(e){this._cubeSize=Math.max(VM,OM(e))}_fromTexture(e,n){e.mapping===Ea||e.mapping===Jo?this._setSize(e.image.length===0?VM:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4);let i=n||this._allocateTarget();return this._textureToCubemap(e),this._applyPMREM(i),i}_allocateTarget(){let e=this._cubeSize,n=kM(e,!1,!1),i=Math.log2(e)-FR;for(let s=0;s<=i;s++)n.texture.mipmaps.push({width:e>>s,height:e>>s});return n.texture.name="PMREM",n.texture.isPMREMTexture=!0,n}_getSourceTarget(e=!1){let n=this._cubeSize,i=this._sourceTarget;return(i===null||i.width!==n||e&&i.depthBuffer===!1)&&(i!==null&&i.dispose(),this._sourceTarget=kM(n,!0,e)),this._sourceTarget}_compileMaterial(e){this._boxMesh.material=e,this._renderer.compile(this._boxMesh,this._cubeCamera.children[0])}_renderCube(e,n,i){let s=this._boxMesh,r=this._cubeCamera;s.material=i,r.renderTarget=e,r.activeMipmapLevel=n;let a=e.width>>n;e.viewport.set(0,0,a,a),r.update(this._renderer,s),e.viewport.set(0,0,e.width,e.height)}_textureToCubemap(e){let n;e.mapping===Ea||e.mapping===Jo?(this._cubemapMaterial===null&&(this._cubemapMaterial=XM()),n=this._cubemapMaterial,n.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):(this._equirectMaterial===null&&(this._equirectMaterial=WM()),n=this._equirectMaterial),n.uniforms.envMap.value=e,this._renderCube(this._getSourceTarget(),0,n)}_applyPMREM(e){this._ggxMaterial===null&&(this._ggxMaterial=kR(),this._integrationMaterial=WR());let n=this._cubeSize,i=e.texture.mipmaps.length-1,s=this._ggxMaterial.uniforms;s.envMap.value=this._sourceTarget.texture;let r=this._integrationMaterial.uniforms;r.envMap.value=this._sourceTarget.texture,r.sourceLod.value=Math.log2(n/h1);for(let a=0;a<=i;a++){let o=t.lodToRoughness(a,i);if(a>i-HR)r.roughness.value=o,this._renderCube(e,a,this._integrationMaterial);else{let l=o>0?Math.log2(n)+.5*Math.log2(6/(f1*Math.pow(o,4)))+.5:0;s.roughness.value=o,s.lodBias.value=l,this._renderCube(e,a,this._ggxMaterial)}}}_blur(e,n){this._blurMaterial===null&&(this._blurMaterial=XR());let i=this._blurMaterial,s=i.uniforms,r=this._sourceTarget;s.sigma.value=Math.min(n,Math.PI)/Math.SQRT2,s.envMap.value=r.texture,this._renderCube(e,0,i),s.envMap.value=e.texture,this._renderCube(r,0,i)}};function kM(t,e,n){return new Pc(t,{minFilter:Xs,generateMipmaps:e,type:Xi,colorSpace:Ws,depthBuffer:n})}function Oc(t,e,n){return new hn({name:t,uniforms:e,vertexShader:Ve.cube_vert,fragmentShader:n,side:Qt,blending:ki,depthTest:!1,depthWrite:!1})}function kR(){return Oc("PMREMGGXConvolution",{envMap:{value:null},roughness:{value:0},lodBias:{value:0}},`

		#define GGX_SAMPLES ${f1}u

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float roughness;
		uniform float lodBias;

		#include <common>

		// Van der Corput radical inverse
		float radicalInverse_VdC( uint bits ) {

			bits = ( bits << 16u ) | ( bits >> 16u );
			bits = ( ( bits & 0x55555555u ) << 1u ) | ( ( bits & 0xAAAAAAAAu ) >> 1u );
			bits = ( ( bits & 0x33333333u ) << 2u ) | ( ( bits & 0xCCCCCCCCu ) >> 2u );
			bits = ( ( bits & 0x0F0F0F0Fu ) << 4u ) | ( ( bits & 0xF0F0F0F0u ) >> 4u );
			bits = ( ( bits & 0x00FF00FFu ) << 8u ) | ( ( bits & 0xFF00FF00u ) >> 8u );
			return float( bits ) * 2.3283064365386963e-10; // / 0x100000000

		}

		// Hammersley sequence
		vec2 hammersley( uint i, uint N ) {

			return vec2( float( i ) / float( N ), radicalInverse_VdC( i ) );

		}

		void main() {

			vec3 N = normalize( vWorldDirection );

			// For very low roughness, just sample the environment directly
			if ( roughness < 0.001 ) {

				gl_FragColor = vec4( textureLod( envMap, N, 0.0 ).rgb, 1.0 );
				return;

			}

			float alpha = roughness * roughness;
			float alpha2 = alpha * alpha;

			// Tangent space basis for VNDF sampling
			vec3 up = abs( N.z ) < 0.999 ? vec3( 0.0, 0.0, 1.0 ) : vec3( 1.0, 0.0, 0.0 );
			vec3 tangent = normalize( cross( up, N ) );
			vec3 bitangent = cross( N, tangent );

			vec3 prefilteredColor = vec3( 0.0 );
			float totalWeight = 0.0;

			for ( uint i = 0u; i < GGX_SAMPLES; i ++ ) {

				vec2 Xi = hammersley( i, GGX_SAMPLES );

				// With V = N, sample the reflected direction directly.
				float invQ = 1.0 / ( 1.0 - Xi.x + alpha2 * Xi.x );
				float NdotL = ( 1.0 - Xi.x - alpha2 * Xi.x ) * invQ;

				if ( NdotL > 0.0 ) {

					float phi = 2.0 * PI * Xi.y;
					float sinTheta = 2.0 * alpha * sqrt( Xi.x * ( 1.0 - Xi.x ) ) * invQ;
					vec3 L = N * NdotL + ( tangent * cos( phi ) + bitangent * sin( phi ) ) * sinTheta;

					// Match the source mip to the sample's solid angle. The terms
					// independent of the sample are precomputed in lodBias.
					float d = alpha2 * invQ;
					float lod = max( log2( d ) + lodBias, 0.0 );

					// Weight by NdotL for the split-sum approximation
					prefilteredColor += textureLod( envMap, L, lod ).rgb * NdotL;
					totalWeight += NdotL;

				}

			}

			gl_FragColor = vec4( prefilteredColor / totalWeight, 1.0 );

		}
	`)}function WR(){return Oc("PMREMGGXIntegration",{envMap:{value:null},roughness:{value:0},sourceLod:{value:0},sourceSize:{value:h1}},`

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float roughness;
		uniform float sourceLod;
		uniform int sourceSize;

		void main() {

			vec3 N = normalize( vWorldDirection );

			float alpha = roughness * roughness;
			float alpha2 = alpha * alpha;

			float texelSize = 2.0 / float( sourceSize );

			vec3 prefilteredColor = vec3( 0.0 );
			float totalWeight = 0.0;

			// Pair opposite texels: only the one in N's hemisphere contributes.
			for ( int face = 0; face < 3; face ++ ) {

				for ( int y = 0; y < sourceSize; y ++ ) {

					for ( int x = 0; x < sourceSize; x ++ ) {

						vec2 uv = ( vec2( x, y ) + 0.5 ) * texelSize - 1.0;
						vec3 texelDirection = face == 0 ? vec3( 1.0, uv ) : ( face == 1 ? vec3( uv.x, 1.0, uv.y ) : vec3( uv, 1.0 ) );

						float invDistance = inversesqrt( 1.0 + dot( uv, uv ) );
						float NdotL = dot( N, texelDirection );
						texelDirection *= NdotL < 0.0 ? - 1.0 : 1.0;
						NdotL = abs( NdotL ) * invDistance;

						// With V = N, NdotH squared is ( 1 + NdotL ) / 2. Common factors
						// in the GGX distribution and texel solid angle cancel when normalized.
						float d = 1.0 + alpha2 + ( alpha2 - 1.0 ) * NdotL;
						float weight = NdotL * invDistance * invDistance * invDistance / ( d * d );

						prefilteredColor += textureLod( envMap, texelDirection, sourceLod ).rgb * weight;
						totalWeight += weight;

					}

				}

			}

			gl_FragColor = vec4( prefilteredColor / totalWeight, 1.0 );

		}
	`)}function XR(){return Oc("PMREMSphericalGaussianBlur",{envMap:{value:null},sigma:{value:0}},`

		#define SAMPLES ${zR}
		#define GOLDEN_ANGLE 2.39996322973

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float sigma;

		#include <common>

		void main() {

			vec3 outputDirection = normalize( vWorldDirection );

			vec3 up = abs( outputDirection.z ) < 0.999 ? vec3( 0.0, 0.0, 1.0 ) : vec3( 1.0, 0.0, 0.0 );
			vec3 tangent = normalize( cross( up, outputDirection ) );
			vec3 bitangent = cross( outputDirection, tangent );

			// Truncate the kernel at three standard deviations or at the antipode.
			float thetaMax = min( 3.0 * sigma, PI );
			float truncation = 1.0 - exp( - 0.5 * thetaMax * thetaMax / ( sigma * sigma ) );

			vec3 accumColor = vec3( 0.0 );
			float accumWeight = 0.0;

			for ( int i = 0; i < SAMPLES; i ++ ) {

				// Stratified inverse-CDF sampling of the Gaussian, placed on a golden-angle spiral.
				float stratum = ( float( i ) + 0.5 ) / float( SAMPLES );
				float theta = sigma * sqrt( - 2.0 * log( 1.0 - stratum * truncation ) );
				float phi = float( i ) * GOLDEN_ANGLE;

				vec3 offset = cos( phi ) * tangent + sin( phi ) * bitangent;
				vec3 sampleDirection = cos( theta ) * outputDirection + sin( theta ) * offset;

				// Correct the planar sample density to solid angle.
				float weight = sin( theta ) / theta;

				accumColor += weight * textureLod( envMap, sampleDirection, 0.0 ).rgb;
				accumWeight += weight;

			}

			gl_FragColor = vec4( accumColor / accumWeight, 1.0 );

		}
	`)}function WM(){return Oc("PMREMEquirectangularToCubemap",{envMap:{value:null}},`

		varying vec3 vWorldDirection;

		uniform sampler2D envMap;

		#include <common>

		void main() {

			// Average four subpixel samples to preserve small, bright features.
			vec3 direction = normalize( vWorldDirection );
			vec3 dx = dFdx( direction ) * 0.25;
			vec3 dy = dFdy( direction ) * 0.25;

			vec3 color = textureLod( envMap, equirectUv( normalize( direction - dx - dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction + dx - dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction - dx + dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction + dx + dy ) ), 0.0 ).rgb;
			gl_FragColor = vec4( color * 0.25, 1.0 );

		}
	`)}function XM(){return Oc("PMREMCubemapToCubemap",{envMap:{value:null},flipEnvMap:{value:-1}},`

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float flipEnvMap;

		void main() {

			gl_FragColor = vec4( textureCube( envMap, vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) ).rgb, 1.0 );

		}
	`)}function YR(t){let e=new WeakMap,n=new WeakMap,i=null;function s(u,d=!1){return u==null?null:d?a(u):r(u)}function r(u){if(u&&u.isTexture){let d=u.mapping;if(d===Ch||d===Rh)if(e.has(u)){let x=e.get(u).texture;return o(x,u.mapping)}else{let x=u.image;if(x&&x.height>0){let A=new Pc(x.height);return A.fromEquirectangularTexture(t,u),e.set(u,A),u.addEventListener("dispose",c),o(A.texture,u.mapping)}else return null}}return u}function a(u){if(u&&u.isTexture&&u.isPMREMTexture!==!0){let d=u.mapping,x=d===Ch||d===Rh,A=d===Ea||d===Jo;if(x||A){let g=n.get(u),f=g!==void 0?g.texture.pmremVersion:0;if(u.isRenderTargetTexture&&u.pmremVersion!==f)return i===null&&(i=new md(t)),g=x?i.fromEquirectangular(u,g):i.fromCubemap(u,g),g.texture.pmremVersion=u.pmremVersion,n.set(u,g),g.texture;if(g!==void 0)return g.texture;{let m=u.image;return x&&m&&m.height>0||A&&m&&l(m)?(i===null&&(i=new md(t)),g=x?i.fromEquirectangular(u):i.fromCubemap(u),g.texture.pmremVersion=u.pmremVersion,n.set(u,g),u.addEventListener("dispose",h),g.texture):null}}}return u}function o(u,d){return d===Ch?u.mapping=Ea:d===Rh&&(u.mapping=Jo),u}function l(u){let d=0,x=6;for(let A=0;A<x;A++)u[A]!==void 0&&d++;return d===x}function c(u){let d=u.target;d.removeEventListener("dispose",c);let x=e.get(d);x!==void 0&&(e.delete(d),x.dispose())}function h(u){let d=u.target;d.removeEventListener("dispose",h);let x=n.get(d);x!==void 0&&(n.delete(d),x.dispose())}function p(){e=new WeakMap,n=new WeakMap,i!==null&&(i.dispose(),i=null)}return{get:s,dispose:p}}function qR(t){let e={};function n(i){if(e[i]!==void 0)return e[i];let s=t.getExtension(i);return e[i]=s,s}return{has:function(i){return n(i)!==null},init:function(){n("EXT_color_buffer_float"),n("WEBGL_clip_cull_distance"),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float"),n("WEBGL_multisampled_render_to_texture"),n("WEBGL_render_shared_exponent")},get:function(i){let s=n(i);return s===null&&cc("WebGLRenderer: "+i+" extension not supported."),s}}}function QR(t,e,n,i){let s={},r=new WeakMap,a=new FinalizationRegistry(x=>delete s[x]);function o(x){l(x.target)}function l(x){x.index!==null&&e.remove(x.index);for(let g in x.attributes)e.remove(x.attributes[g]);x.removeEventListener("dispose",o),delete s[x.id],a.unregister(x);let A=r.get(x);A&&(e.remove(A),r.delete(x)),i.releaseStatesOfGeometry(x),x.isInstancedBufferGeometry===!0&&delete x._maxInstanceCount,n.memory.geometries--}function c(x,A){return s[A.id]!==void 0||(A.addEventListener("dispose",o),s[A.id]=new WeakRef(A),a.register(A,A.id,A),n.memory.geometries++),A}function h(x){let A=x.attributes;for(let g in A)e.update(A[g],t.ARRAY_BUFFER)}function p(x){let A=x.index,g=x.attributes.position,f=0;if(g===void 0)return;let m=A!==null?A.array.length:g.array.length/3-1,y=g.count>=65535?vc:gc,_=new y(Math.ceil(m/3)*6,1),T=_.array;if(A!==null){let v=A.array;f=A.version;for(let E=0,N=0;E<m;E+=3){let C=v[E+0],B=v[E+1],F=v[E+2];T[N++]=C,T[N++]=B,T[N++]=B,T[N++]=F,T[N++]=F,T[N++]=C}}else{f=g.version;for(let v=0,E=0;v<m;v+=3){let N=v+0,C=v+1,B=v+2;T[E++]=N,T[E++]=C,T[E++]=C,T[E++]=B,T[E++]=B,T[E++]=N}}_.version=f;let b=r.get(x);b&&e.remove(b),r.set(x,_)}function u(x){let A=r.get(x);if(A){let g=x.index;g!==null&&A.version<g.version&&p(x)}else p(x);return r.get(x)}function d(){for(let x in s){let A=s[x].deref();A!==void 0&&l(A)}}return{get:c,update:h,getWireframeAttribute:u,dispose:d}}function ZR(t,e,n){let i;function s(p){i=p}let r,a;function o(p){r=p.type,a=p.bytesPerElement}function l(p,u){t.drawElements(i,u,r,p*a),n.update(u,i,1)}function c(p,u,d){d!==0&&(t.drawElementsInstanced(i,u,r,p*a,d),n.update(u,i,d))}function h(p,u,d){if(d===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,u,0,r,p,0,d);let A=0;for(let g=0;g<d;g++)A+=u[g];n.update(A,i,1)}this.setMode=s,this.setIndex=o,this.render=l,this.renderInstances=c,this.renderMultiDraw=h}function KR(t){let e={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,a,o){switch(n.calls++,a){case t.TRIANGLES:n.triangles+=o*(r/3);break;case t.LINES:n.lines+=o*(r/2);break;case t.LINE_STRIP:n.lines+=o*(r-1);break;case t.LINE_LOOP:n.lines+=o*r;break;case t.POINTS:n.points+=o*r;break;default:Be("WebGLInfo: Unknown draw mode:",a);break}}function s(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:e,render:n,programs:null,autoReset:!0,reset:s,update:i}}function JR(t,e,n){let i=new WeakMap,s=new Lt;function r(a,o,l){let c=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,p=h!==void 0?h.length:0,u=i.get(o);if(u===void 0||u.count!==p){let N=function(){v.dispose(),i.delete(o),o.removeEventListener("dispose",N)};u!==void 0&&u.texture.dispose();let d=o.morphAttributes.position!==void 0,x=o.morphAttributes.normal!==void 0,A=o.morphAttributes.color!==void 0,g=o.morphAttributes.position||[],f=o.morphAttributes.normal||[],m=o.morphAttributes.color||[],y=0;d===!0&&(y=1),x===!0&&(y=2),A===!0&&(y=3);let _=o.attributes.position.count*y,T=1;_>e.maxTextureSize&&(T=Math.ceil(_/e.maxTextureSize),_=e.maxTextureSize);let b=new Float32Array(_*T*4*p),v=new pc(b,_,T,p);v.type=fi,v.needsUpdate=!0;let E=y*4;for(let C=0;C<p;C++){let B=g[C],F=f[C],X=m[C],D=_*T*4*C;for(let V=0;V<B.count;V++){let q=V*E;d===!0&&(s.fromBufferAttribute(B,V),b[D+q+0]=s.x,b[D+q+1]=s.y,b[D+q+2]=s.z,b[D+q+3]=0),x===!0&&(s.fromBufferAttribute(F,V),b[D+q+4]=s.x,b[D+q+5]=s.y,b[D+q+6]=s.z,b[D+q+7]=0),A===!0&&(s.fromBufferAttribute(X,V),b[D+q+8]=s.x,b[D+q+9]=s.y,b[D+q+10]=s.z,b[D+q+11]=X.itemSize===4?s.w:1)}}u={count:p,texture:v,size:new Oe(_,T)},i.set(o,u),o.addEventListener("dispose",N)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)l.getUniforms().setValue(t,"morphTexture",a.morphTexture,n);else{let d=0;for(let A=0;A<c.length;A++)d+=c[A];let x=o.morphTargetsRelative?1:1-d;l.getUniforms().setValue(t,"morphTargetBaseInfluence",x),l.getUniforms().setValue(t,"morphTargetInfluences",c)}l.getUniforms().setValue(t,"morphTargetsTexture",u.texture,n),l.getUniforms().setValue(t,"morphTargetsTextureSize",u.size)}return{update:r}}function jR(t,e,n,i,s){let r=new WeakMap;function a(c){let h=s.render.frame,p=c.geometry,u=e.get(c,p);if(r.get(u)!==h&&(e.update(u),r.set(u,h)),c.isInstancedMesh&&(c.hasEventListener("dispose",l)===!1&&c.addEventListener("dispose",l),r.get(c)!==h&&(n.update(c.instanceMatrix,t.ARRAY_BUFFER),c.instanceColor!==null&&n.update(c.instanceColor,t.ARRAY_BUFFER),r.set(c,h))),c.isSkinnedMesh){let d=c.skeleton;r.get(d)!==h&&(d.update(),r.set(d,h))}return u}function o(){r=new WeakMap}function l(c){let h=c.target;h.removeEventListener("dispose",l),i.releaseStatesOfObject(h),n.remove(h.instanceMatrix),h.instanceColor!==null&&n.remove(h.instanceColor)}return{update:a,dispose:o}}var $R={[lv]:"LINEAR_TONE_MAPPING",[cv]:"REINHARD_TONE_MAPPING",[uv]:"CINEON_TONE_MAPPING",[fv]:"ACES_FILMIC_TONE_MAPPING",[dv]:"AGX_TONE_MAPPING",[pv]:"NEUTRAL_TONE_MAPPING",[hv]:"CUSTOM_TONE_MAPPING"};function eD(t,e,n,i,s,r){let a=new zt(e,n,{type:t,depthBuffer:s,stencilBuffer:r,samples:i?4:0,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,resolveDepthBuffer:!1,resolveStencilBuffer:!1}),o=null,l=null,c=null,h=new ps;h.setAttribute("position",new Ri([-1,3,0,-1,-1,0,3,-1,0],3)),h.setAttribute("uv",new Ri([0,2,0,0,2,0],2));let p=new dh({uniforms:{tDiffuse:{value:null}},vertexShader:`
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
			}`,depthTest:!1,depthWrite:!1}),u=new Qn(h,p),d=new Fr(-1,1,1,-1,0,1),x=null,A=null,g=!1,f,m=null,y=[],_=!1;this.setSize=function(T,b){a.setSize(T,b),o!==null&&o.setSize(T,b),l!==null&&l.setSize(T,b);for(let v=0;v<y.length;v++){let E=y[v];E.setSize&&E.setSize(T,b)}},this.setEffects=function(T){y=T,_=y.length>0&&y[0].isRenderPass===!0;let b=a.width,v=a.height;y.length>0&&o===null&&(o=new zt(b,v,{type:Xi,depthBuffer:!1,stencilBuffer:!1}),l=new zt(b,v,{type:Xi,depthBuffer:!1,stencilBuffer:!1}));let E=s===!0&&y.some(N=>N.needsDepthTexture===!0);E!==(a.depthTexture!==null)&&(E===!0&&c===null&&(c=new ci(b,v),r===!0&&(c.format=qi,c.type=Ys)),a.dispose(),a.depthTexture=E?c:null,a.resolveDepthBuffer=E,a.storeMultisampledDepthBuffer=E);for(let N=0;N<y.length;N++){let C=y[N];C.setSize&&C.setSize(b,v),E&&C.needsDepthTexture===!0&&C.setDepthTexture&&C.setDepthTexture(c)}},this.begin=function(T,b){if(g||T.toneMapping===gs&&y.length===0)return!1;if(m=b,b!==null){let v=b.width,E=b.height;(a.width!==v||a.height!==E)&&this.setSize(v,E)}return _===!1&&T.setRenderTarget(a),f=T.toneMapping,T.toneMapping=gs,!0},this.hasRenderPass=function(){return _},this.end=function(T,b){T.toneMapping=f,g=!0;let v=a,E=o;for(let N=0;N<y.length;N++){let C=y[N];C.enabled!==!1&&(C.render(T,E,v,b),C.needsSwap!==!1&&(v=E,E=E===o?l:o))}if(x!==T.outputColorSpace||A!==T.toneMapping){x=T.outputColorSpace,A=T.toneMapping,p.defines={},je.getTransfer(x)===ct&&(p.defines.SRGB_TRANSFER="");let N=$R[A];N&&(p.defines[N]=""),p.needsUpdate=!0}p.uniforms.tDiffuse.value=v.texture,T.setRenderTarget(m),T.render(u,d),m=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){a.dispose(),o!==null&&o.dispose(),l!==null&&l.dispose(),h.dispose(),p.dispose()}}var d1=new qt,p1=new ci(1,1),m1=new yc(1),g1=new pc,v1=new lh,x1=new xc,YM=[],qM=[],QM=new Float32Array(16),ZM=new Float32Array(9),KM=new Float32Array(4);function el(t,e,n){let i=t[0];if(i<=0||i>0)return t;let s=e*n,r=YM[s];if(r===void 0&&(r=new Float32Array(s),YM[s]=r),e!==0){i.toArray(r,0);for(let a=1,o=0;a!==e;++a)o+=n,t[a].toArray(r,o)}return r}function en(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function tn(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function xd(t,e){let n=qM[e];n===void 0&&(n=new Int32Array(e),qM[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function tD(t,e){let n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function nD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(en(n,e))return;t.uniform2fv(this.addr,e),tn(n,e)}}function iD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(e.r!==void 0)(n[0]!==e.r||n[1]!==e.g||n[2]!==e.b)&&(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(en(n,e))return;t.uniform3fv(this.addr,e),tn(n,e)}}function sD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(en(n,e))return;t.uniform4fv(this.addr,e),tn(n,e)}}function rD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(en(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),tn(n,e)}else{if(en(n,i))return;KM.set(i),t.uniformMatrix2fv(this.addr,!1,KM),tn(n,i)}}function aD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(en(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),tn(n,e)}else{if(en(n,i))return;ZM.set(i),t.uniformMatrix3fv(this.addr,!1,ZM),tn(n,i)}}function oD(t,e){let n=this.cache,i=e.elements;if(i===void 0){if(en(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),tn(n,e)}else{if(en(n,i))return;QM.set(i),t.uniformMatrix4fv(this.addr,!1,QM),tn(n,i)}}function lD(t,e){let n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function cD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2i(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(en(n,e))return;t.uniform2iv(this.addr,e),tn(n,e)}}function uD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3i(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(en(n,e))return;t.uniform3iv(this.addr,e),tn(n,e)}}function fD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4i(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(en(n,e))return;t.uniform4iv(this.addr,e),tn(n,e)}}function hD(t,e){let n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function dD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2ui(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(en(n,e))return;t.uniform2uiv(this.addr,e),tn(n,e)}}function pD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3ui(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(en(n,e))return;t.uniform3uiv(this.addr,e),tn(n,e)}}function mD(t,e){let n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4ui(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(en(n,e))return;t.uniform4uiv(this.addr,e),tn(n,e)}}function yd(t){return t.version===0&&(t.compareFunction=bv,t.needsUpdate=!0),t}function gD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s);let r;this.type===t.SAMPLER_2D_SHADOW?r=yd(p1):r=d1,n.setTexture2D(e||r,s)}function vD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture3D(e||v1,s)}function xD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s);let r;this.type===t.SAMPLER_CUBE_SHADOW?r=yd(m1):r=x1,n.setTextureCube(e||r,s)}function yD(t,e,n){let i=this.cache,s=n.allocateTextureUnit();i[0]!==s&&(t.uniform1i(this.addr,s),i[0]=s),n.setTexture2DArray(e||g1,s)}function _D(t){switch(t){case 5126:return tD;case 35664:return nD;case 35665:return iD;case 35666:return sD;case 35674:return rD;case 35675:return aD;case 35676:return oD;case 5124:case 35670:return lD;case 35667:case 35671:return cD;case 35668:case 35672:return uD;case 35669:case 35673:return fD;case 5125:return hD;case 36294:return dD;case 36295:return pD;case 36296:return mD;case 35678:case 36198:case 36298:case 36306:case 35682:return gD;case 35679:case 36299:case 36307:return vD;case 35680:case 36300:case 36308:case 36293:return xD;case 36289:case 36303:case 36311:case 36292:return yD}}function SD(t,e){t.uniform1fv(this.addr,e)}function AD(t,e){let n=el(e,this.size,2);t.uniform2fv(this.addr,n)}function MD(t,e){let n=el(e,this.size,3);t.uniform3fv(this.addr,n)}function ED(t,e){let n=el(e,this.size,4);t.uniform4fv(this.addr,n)}function TD(t,e){let n=el(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function bD(t,e){let n=el(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function wD(t,e){let n=el(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function CD(t,e){t.uniform1iv(this.addr,e)}function RD(t,e){t.uniform2iv(this.addr,e)}function DD(t,e){t.uniform3iv(this.addr,e)}function UD(t,e){t.uniform4iv(this.addr,e)}function ID(t,e){t.uniform1uiv(this.addr,e)}function BD(t,e){t.uniform2uiv(this.addr,e)}function ND(t,e){t.uniform3uiv(this.addr,e)}function LD(t,e){t.uniform4uiv(this.addr,e)}function PD(t,e,n){let i=this.cache,s=e.length,r=xd(n,s);en(i,r)||(t.uniform1iv(this.addr,r),tn(i,r));let a;this.type===t.SAMPLER_2D_SHADOW?a=yd(p1):a=d1;for(let o=0;o!==s;++o)n.setTexture2D(e[o]||a,r[o])}function OD(t,e,n){let i=this.cache,s=e.length,r=xd(n,s);en(i,r)||(t.uniform1iv(this.addr,r),tn(i,r));for(let a=0;a!==s;++a)n.setTexture3D(e[a]||v1,r[a])}function FD(t,e,n){let i=this.cache,s=e.length,r=xd(n,s);en(i,r)||(t.uniform1iv(this.addr,r),tn(i,r));let a;this.type===t.SAMPLER_CUBE_SHADOW?a=yd(m1):a=x1;for(let o=0;o!==s;++o)n.setTextureCube(e[o]||a,r[o])}function zD(t,e,n){let i=this.cache,s=e.length,r=xd(n,s);en(i,r)||(t.uniform1iv(this.addr,r),tn(i,r));for(let a=0;a!==s;++a)n.setTexture2DArray(e[a]||g1,r[a])}function HD(t){switch(t){case 5126:return SD;case 35664:return AD;case 35665:return MD;case 35666:return ED;case 35674:return TD;case 35675:return bD;case 35676:return wD;case 5124:case 35670:return CD;case 35667:case 35671:return RD;case 35668:case 35672:return DD;case 35669:case 35673:return UD;case 5125:return ID;case 36294:return BD;case 36295:return ND;case 36296:return LD;case 35678:case 36198:case 36298:case 36306:case 35682:return PD;case 35679:case 36299:case 36307:return OD;case 35680:case 36300:case 36308:case 36293:return FD;case 36289:case 36303:case 36311:case 36292:return zD}}var Hv=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.setValue=_D(n.type)}},Gv=class{constructor(e,n,i){this.id=e,this.addr=i,this.cache=[],this.type=n.type,this.size=n.size,this.setValue=HD(n.type)}},Vv=class{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,n,i){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(e,n[o.id],i)}}},Fv=/(\w+)(\])?(\[|\.)?/g;function JM(t,e){t.seq.push(e),t.map[e.id]=e}function GD(t,e,n){let i=t.name,s=i.length;for(Fv.lastIndex=0;;){let r=Fv.exec(i),a=Fv.lastIndex,o=r[1],l=r[2]==="]",c=r[3];if(l&&(o=o|0),c===void 0||c==="["&&a+2===s){JM(n,c===void 0?new Hv(o,t,e):new Gv(o,t,e));break}else{let p=n.map[o];p===void 0&&(p=new Vv(o),JM(n,p)),n=p}}}var $o=class{constructor(e,n){this.seq=[],this.map={};let i=e.getProgramParameter(n,e.ACTIVE_UNIFORMS);for(let a=0;a<i;++a){let o=e.getActiveUniform(n,a),l=e.getUniformLocation(n,o.name);GD(o,l,this)}let s=[],r=[];for(let a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(e,n,i,s){let r=this.map[n];r!==void 0&&r.setValue(e,i,s)}setOptional(e,n,i){let s=n[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,n,i,s){for(let r=0,a=n.length;r!==a;++r){let o=n[r],l=i[o.id];l.needsUpdate!==!1&&o.setValue(e,l.value,s)}}static seqWithValue(e,n){let i=[];for(let s=0,r=e.length;s!==r;++s){let a=e[s];a.id in n&&i.push(a)}return i}};function jM(t,e,n){let i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}var VD=37297,kD=0;function WD(t,e){let n=t.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,n.length);for(let a=s;a<r;a++){let o=a+1;i.push(`${o===e?">":" "} ${o}: ${n[a]}`)}return i.join(`
`)}var $M=new Ne;function XD(t){je._getMatrix($M,je.workingColorSpace,t);let e=`mat3( ${$M.elements.map(n=>n.toFixed(4))} )`;switch(je.getTransfer(t)){case fc:return[e,"LinearTransferOETF"];case ct:return[e,"sRGBTransferOETF"];default:return De("WebGLProgram: Unsupported color space: ",t),[e,"LinearTransferOETF"]}}function e1(t,e,n){let i=t.getShaderParameter(e,t.COMPILE_STATUS),r=(t.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return n.toUpperCase()+`

`+r+`

`+WD(t.getShaderSource(e),o)}else return r}function YD(t,e){let n=XD(e);return[`vec4 ${t}( vec4 value ) {`,`	return ${n[1]}( vec4( ${n[0]} * value.rgb, value.a ) );`,"}"].join(`
`)}var qD={[lv]:"Linear",[cv]:"Reinhard",[uv]:"Cineon",[fv]:"ACESFilmic",[dv]:"AgX",[pv]:"Neutral",[hv]:"Custom"};function QD(t,e){let n=qD[e];return n===void 0?(De("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+t+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}var pd=new G;function ZD(){je.getLuminanceCoefficients(pd);let t=pd.x.toFixed(4),e=pd.y.toFixed(4),n=pd.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${t}, ${e}, ${n} );`,"	return dot( weights, rgb );","}"].join(`
`)}function KD(t){return[t.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",t.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Lc).join(`
`)}function JD(t){let e=[];for(let n in t){let i=t[n];i!==!1&&e.push("#define "+n+" "+i)}return e.join(`
`)}function jD(t,e){let n={},i=t.getProgramParameter(e,t.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){let r=t.getActiveAttrib(e,s),a=r.name,o=1;r.type===t.FLOAT_MAT2&&(o=2),r.type===t.FLOAT_MAT3&&(o=3),r.type===t.FLOAT_MAT4&&(o=4),n[a]={type:r.type,location:t.getAttribLocation(e,a),locationSize:o}}return n}function Lc(t){return t!==""}function t1(t,e){let n=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return t.replace(/NUM_SUN_LIGHTS/g,e.numSunLights).replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_SUN_LIGHT_SHADOWS/g,e.numSunLightShadows).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows).replace(/NUM_LIGHT_PROBE_GRIDS/g,e.numLightProbeGrids)}function n1(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var $D=/^[ \t]*#include +<([\w\d./]+)>/gm;function kv(t){return t.replace($D,e3)}function e3(t,e){let n=Ve[e];if(n===void 0)throw new Error("THREE.WebGLProgram: Can not resolve #include <"+e+">");return kv(n)}var t3=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function i1(t){return t.replace(t3,n3)}function n3(t,e,n,i){let s="";for(let r=parseInt(e);r<parseInt(n);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function s1(t){let e=`precision ${t.precision} float;
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
#define LOW_PRECISION`),e}var i3={[Ec]:"SHADOWMAP_TYPE_PCF",[Zo]:"SHADOWMAP_TYPE_VSM"};function s3(t){return i3[t.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var r3={[Jo]:"ENVMAP_MODE_REFRACTION"};function a3(t){return t.envMap===!1?"ENVMAP_MODE_REFLECTION":r3[t.envMapMode]||"ENVMAP_MODE_REFLECTION"}var o3={[ov]:"ENVMAP_BLENDING_MULTIPLY",[AM]:"ENVMAP_BLENDING_MIX",[MM]:"ENVMAP_BLENDING_ADD"};function l3(t){return t.envMap===!1?"ENVMAP_BLENDING_NONE":o3[t.combine]||"ENVMAP_BLENDING_NONE"}function c3(t,e,n,i){let s=t.getContext(),r=n.defines,a=n.vertexShader,o=n.fragmentShader,l=s3(n),c=n.envMapPMREM===!0?"ENVMAP_TYPE_PMREM":"ENVMAP_TYPE_CUBE",h=a3(n),p=l3(n),u=KD(n),d=JD(r),x=s.createProgram(),A,g,f=n.glslVersion?"#version "+n.glslVersion+`
`:"";n.isRawShaderMaterial?(A=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,d].filter(Lc).join(`
`),A.length>0&&(A+=`
`),g=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,d].filter(Lc).join(`
`),g.length>0&&(g+=`
`)):(A=[s1(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,d,n.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",n.batching?"#define USE_BATCHING":"",n.batchingColor?"#define USE_BATCHING_COLOR":"",n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.instancingMorph?"#define USE_INSTANCING_MORPH":"",n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+h:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.displacementMap?"#define USE_DISPLACEMENTMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.diffuseRoughnessMap?"#define USE_DIFFUSE_ROUGHNESSMAP":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.mapUv?"#define MAP_UV "+n.mapUv:"",n.alphaMapUv?"#define ALPHAMAP_UV "+n.alphaMapUv:"",n.lightMapUv?"#define LIGHTMAP_UV "+n.lightMapUv:"",n.aoMapUv?"#define AOMAP_UV "+n.aoMapUv:"",n.emissiveMapUv?"#define EMISSIVEMAP_UV "+n.emissiveMapUv:"",n.bumpMapUv?"#define BUMPMAP_UV "+n.bumpMapUv:"",n.normalMapUv?"#define NORMALMAP_UV "+n.normalMapUv:"",n.displacementMapUv?"#define DISPLACEMENTMAP_UV "+n.displacementMapUv:"",n.metalnessMapUv?"#define METALNESSMAP_UV "+n.metalnessMapUv:"",n.roughnessMapUv?"#define ROUGHNESSMAP_UV "+n.roughnessMapUv:"",n.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+n.anisotropyMapUv:"",n.clearcoatMapUv?"#define CLEARCOATMAP_UV "+n.clearcoatMapUv:"",n.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+n.clearcoatNormalMapUv:"",n.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+n.clearcoatRoughnessMapUv:"",n.diffuseRoughnessMapUv?"#define DIFFUSE_ROUGHNESSMAP_UV "+n.diffuseRoughnessMapUv:"",n.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+n.iridescenceMapUv:"",n.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+n.iridescenceThicknessMapUv:"",n.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+n.sheenColorMapUv:"",n.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+n.sheenRoughnessMapUv:"",n.specularMapUv?"#define SPECULARMAP_UV "+n.specularMapUv:"",n.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+n.specularColorMapUv:"",n.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+n.specularIntensityMapUv:"",n.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+n.transmissionMapUv:"",n.thicknessMapUv?"#define THICKNESSMAP_UV "+n.thicknessMapUv:"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexNormals?"#define HAS_NORMAL":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&n.flatShading===!1?"#define USE_MORPHNORMALS":"",n.morphColors?"#define USE_MORPHCOLORS":"",n.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+n.morphTextureStride:"",n.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Lc).join(`
`),g=[s1(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,d,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+c:"",n.envMap?"#define "+h:"",n.envMap?"#define "+p:"",n.envMapPMREM?"#define ENVMAP_MAX_LOD "+n.envMapMaxLod+".0":"",n.envMapPMREM?"#define ENVMAP_SIZE "+n.envMapSize+".0":"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.diffuseRoughness?"#define USE_DIFFUSE_ROUGHNESS":"",n.diffuseRoughnessMap?"#define USE_DIFFUSE_ROUGHNESSMAP":"",n.dispersion?"#define USE_DISPERSION":"",n.retroreflection?"#define USE_RETROREFLECTION":"",n.iridescence?"#define USE_IRIDESCENCE":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas||n.batchingColor?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+l:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",n.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",n.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",n.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",n.toneMapping!==gs?"#define TONE_MAPPING":"",n.toneMapping!==gs?Ve.tonemapping_pars_fragment:"",n.toneMapping!==gs?QD("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.opaque?"#define OPAQUE":"",Ve.colorspace_pars_fragment,YD("linearToOutputTexel",n.outputColorSpace),ZD(),n.useDepthPacking?"#define DEPTH_PACKING "+n.depthPacking:"",`
`].filter(Lc).join(`
`)),a=kv(a),a=t1(a,n),a=n1(a,n),o=kv(o),o=t1(o,n),o=n1(o,n),a=i1(a),o=i1(o),n.isRawShaderMaterial!==!0&&(f=`#version 300 es
`,A=[u,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+A,g=["#define varying in",n.glslVersion===Ic?"":"layout(location = 0) out highp vec4 pc_fragColor;",n.glslVersion===Ic?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+g);let m=f+A+a,y=f+g+o,_=jM(s,s.VERTEX_SHADER,m),T=jM(s,s.FRAGMENT_SHADER,y);s.attachShader(x,_),s.attachShader(x,T),n.index0AttributeName!==void 0?s.bindAttribLocation(x,0,n.index0AttributeName):n.hasPositionAttribute===!0&&s.bindAttribLocation(x,0,"position"),s.linkProgram(x);function b(C){if(t.debug.checkShaderErrors){let B=s.getProgramInfoLog(x)||"",F=s.getShaderInfoLog(_)||"",X=s.getShaderInfoLog(T)||"",D=B.trim(),V=F.trim(),q=X.trim(),H=!0,te=!0;if(s.getProgramParameter(x,s.LINK_STATUS)===!1)if(H=!1,typeof t.debug.onShaderError=="function")t.debug.onShaderError(s,x,_,T);else{let k=e1(s,_,"vertex"),$=e1(s,T,"fragment");Be("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(x,s.VALIDATE_STATUS)+`

Material Name: `+C.name+`
Material Type: `+C.type+`

Program Info Log: `+D+`
`+k+`
`+$)}else D!==""?De("WebGLProgram: Program Info Log:",D):(V===""||q==="")&&(te=!1);te&&(C.diagnostics={runnable:H,programLog:D,vertexShader:{log:V,prefix:A},fragmentShader:{log:q,prefix:g}})}s.deleteShader(_),s.deleteShader(T),v=new $o(s,x),E=jD(s,x)}let v;this.getUniforms=function(){return v===void 0&&b(this),v};let E;this.getAttributes=function(){return E===void 0&&b(this),E};let N=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return N===!1&&(N=s.getProgramParameter(x,VD)),N},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(x),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=kD++,this.cacheKey=e,this.usedTimes=1,this.program=x,this.vertexShader=_,this.fragmentShader=T,this}var u3=0,Wv=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e,n,i){let s=this._getShaderCacheForMaterial(e);return s.has(n)===!1&&(s.add(n),n.usedTimes++),s.has(i)===!1&&(s.add(i),i.usedTimes++),this}remove(e){let n=this.materialCache.get(e);for(let i of n)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderStage(e){return this._getShaderStage(e.vertexShader)}getFragmentShaderStage(e){return this._getShaderStage(e.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){let n=this.materialCache,i=n.get(e);return i===void 0&&(i=new Set,n.set(e,i)),i}_getShaderStage(e){let n=this.shaderCache,i=n.get(e);return i===void 0&&(i=new Xv(e),n.set(e,i)),i}},Xv=class{constructor(e){this.id=u3++,this.code=e,this.usedTimes=0}};function f3(t){return t===zr||t===Dc||t===Uc}function h3(t,e,n,i,s,r){let a=new mc,o=new Wv,l=new Set,c=[],h=new Map,p=i.logarithmicDepthBuffer,u={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function d(v){return l.add(v),v===0?"uv":`uv${v}`}function x(v,E,N,C,B,F){let X=C.fog,D=B.geometry,V=v.isMeshStandardMaterial||v.isMeshLambertMaterial||v.isMeshPhongMaterial?C.environment:null,q=v.isMeshStandardMaterial||v.isMeshLambertMaterial&&!v.envMap||v.isMeshPhongMaterial&&!v.envMap,H=e.get(v.envMap||V,q),te=u[v.type],k=i.precision;v.precision!==null&&(k=i.getMaxPrecision(v.precision),k!==v.precision&&De("WebGLProgram.getParameters:",v.precision,"not supported, using",k,"instead."));let $=D.morphAttributes.position||D.morphAttributes.normal||D.morphAttributes.color,ee=$!==void 0?$.length:0,_e=0;D.morphAttributes.position!==void 0&&(_e=1),D.morphAttributes.normal!==void 0&&(_e=2),D.morphAttributes.color!==void 0&&(_e=3);let Ie,Q,de,xe;if(te){let me=ys[te];Ie=me.vertexShader,Q=me.fragmentShader}else{Ie=v.vertexShader,Q=v.fragmentShader;let me=o.getVertexShaderStage(v),ut=o.getFragmentShaderStage(v);o.update(v,me,ut),de=me.id,xe=ut.id}let Te=t.getRenderTarget(),pe=t.state.buffers.depth.getReversed(),be=B.isInstancedMesh===!0,qe=B.isBatchedMesh===!0,Ce=!!v.map,st=!!v.matcap,mt=!!H,Qe=mt&&H.isPMREMTexture===!0,dn=!!v.aoMap,Zt=!!v.lightMap,zn=!!v.bumpMap&&v.wireframe===!1,It=!!v.normalMap,Qi=!!v.displacementMap,nn=!!v.emissiveMap,di=!!v.metalnessMap,pi=!!v.roughnessMap,mi=v.anisotropy>0,I=v.clearcoat>0,gi=v.diffuseRoughness>0,bt=v.dispersion>0,wt=v.retroreflectivity>0,he=v.iridescence>0,Kt=v.sheen>0,Re=v.transmission>0,Le=mi&&!!v.anisotropyMap,w=I&&!!v.clearcoatMap,S=I&&!!v.clearcoatNormalMap,P=I&&!!v.clearcoatRoughnessMap,ne=gi&&!!v.diffuseRoughnessMap,j=he&&!!v.iridescenceMap,re=he&&!!v.iridescenceThicknessMap,ue=Kt&&!!v.sheenColorMap,Y=Kt&&!!v.sheenRoughnessMap,Z=!!v.specularMap,oe=!!v.specularColorMap,R=!!v.specularIntensityMap,K=Re&&!!v.transmissionMap,W=Re&&!!v.thicknessMap,le=!!v.gradientMap,se=!!v.alphaMap,ae=v.alphaTest>0,Ee=!!v.alphaHash,Ge=!!v.extensions,ie=gs;v.toneMapped&&(Te===null||Te.isXRRenderTarget===!0)&&(ie=t.toneMapping);let we=Qe?H.isCompressedCubeTexture?H.image[0].mipmaps:H.mipmaps:null,He={shaderID:te,shaderType:v.type,shaderName:v.name,vertexShader:Ie,fragmentShader:Q,defines:v.defines,customVertexShaderID:de,customFragmentShaderID:xe,isRawShaderMaterial:v.isRawShaderMaterial===!0,glslVersion:v.glslVersion,precision:k,batching:qe,batchingColor:qe&&B._colorsTexture!==null,instancing:be,instancingColor:be&&B.instanceColor!==null,instancingMorph:be&&B.morphTexture!==null,outputColorSpace:Te===null?t.outputColorSpace:Te.isXRRenderTarget===!0?Te.texture.colorSpace:je.workingColorSpace,alphaToCoverage:!!v.alphaToCoverage,map:Ce,matcap:st,envMap:mt,envMapMode:mt&&H.mapping,envMapPMREM:Qe,envMapMaxLod:Qe?we.length-1:null,envMapSize:Qe?we[0].width:null,aoMap:dn,lightMap:Zt,bumpMap:zn,normalMap:It,displacementMap:Qi,emissiveMap:nn,normalMapObjectSpace:It&&v.normalMapType===TM,normalMapTangentSpace:It&&v.normalMapType===Mv,packedNormalMap:It&&v.normalMapType===Mv&&f3(v.normalMap.format),metalnessMap:di,roughnessMap:pi,anisotropy:mi,anisotropyMap:Le,clearcoat:I,clearcoatMap:w,clearcoatNormalMap:S,clearcoatRoughnessMap:P,diffuseRoughness:gi,diffuseRoughnessMap:ne,dispersion:bt,retroreflection:wt,iridescence:he,iridescenceMap:j,iridescenceThicknessMap:re,sheen:Kt,sheenColorMap:ue,sheenRoughnessMap:Y,specularMap:Z,specularColorMap:oe,specularIntensityMap:R,transmission:Re,transmissionMap:K,thicknessMap:W,gradientMap:le,opaque:v.transparent===!1&&v.blending===Ko&&v.alphaToCoverage===!1,alphaMap:se,alphaTest:ae,alphaHash:Ee,combine:v.combine,mapUv:Ce&&d(v.map.channel),aoMapUv:dn&&d(v.aoMap.channel),lightMapUv:Zt&&d(v.lightMap.channel),bumpMapUv:zn&&d(v.bumpMap.channel),normalMapUv:It&&d(v.normalMap.channel),displacementMapUv:Qi&&d(v.displacementMap.channel),emissiveMapUv:nn&&d(v.emissiveMap.channel),metalnessMapUv:di&&d(v.metalnessMap.channel),roughnessMapUv:pi&&d(v.roughnessMap.channel),anisotropyMapUv:Le&&d(v.anisotropyMap.channel),clearcoatMapUv:w&&d(v.clearcoatMap.channel),clearcoatNormalMapUv:S&&d(v.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:P&&d(v.clearcoatRoughnessMap.channel),diffuseRoughnessMapUv:ne&&d(v.diffuseRoughnessMap.channel),iridescenceMapUv:j&&d(v.iridescenceMap.channel),iridescenceThicknessMapUv:re&&d(v.iridescenceThicknessMap.channel),sheenColorMapUv:ue&&d(v.sheenColorMap.channel),sheenRoughnessMapUv:Y&&d(v.sheenRoughnessMap.channel),specularMapUv:Z&&d(v.specularMap.channel),specularColorMapUv:oe&&d(v.specularColorMap.channel),specularIntensityMapUv:R&&d(v.specularIntensityMap.channel),transmissionMapUv:K&&d(v.transmissionMap.channel),thicknessMapUv:W&&d(v.thicknessMap.channel),alphaMapUv:se&&d(v.alphaMap.channel),vertexTangents:!!D.attributes.tangent&&(It||mi),vertexNormals:!!D.attributes.normal,vertexColors:v.vertexColors,vertexAlphas:v.vertexColors===!0&&!!D.attributes.color&&D.attributes.color.itemSize===4,pointsUvs:B.isPoints===!0&&!!D.attributes.uv&&(Ce||se),fog:!!X,useFog:v.fog===!0,fogExp2:!!X&&X.isFogExp2,flatShading:v.wireframe===!1&&(v.flatShading===!0||D.attributes.normal===void 0&&It===!1&&(v.isMeshLambertMaterial||v.isMeshPhongMaterial||v.isMeshStandardMaterial||v.isMeshPhysicalMaterial)),sizeAttenuation:v.sizeAttenuation===!0,logarithmicDepthBuffer:p,reversedDepthBuffer:pe,skinning:B.isSkinnedMesh===!0,hasPositionAttribute:D.attributes.position!==void 0,morphTargets:D.morphAttributes.position!==void 0,morphNormals:D.morphAttributes.normal!==void 0,morphColors:D.morphAttributes.color!==void 0,morphTargetsCount:ee,morphTextureStride:_e,numSunLights:E.sun.length,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numSunLightShadows:E.sunShadowMap.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numLightProbeGrids:F.filter(me=>me.texture!==null).length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:v.dithering,shadowMapEnabled:t.shadowMap.enabled&&N.length>0,shadowMapType:t.shadowMap.type,toneMapping:ie,decodeVideoTexture:Ce&&v.map.isVideoTexture===!0&&je.getTransfer(v.map.colorSpace)===ct,decodeVideoTextureEmissive:nn&&v.emissiveMap.isVideoTexture===!0&&je.getTransfer(v.emissiveMap.colorSpace)===ct,premultipliedAlpha:v.premultipliedAlpha,doubleSided:v.side===Fn,flipSided:v.side===Qt,useDepthPacking:v.depthPacking>=0,depthPacking:v.depthPacking||0,index0AttributeName:v.index0AttributeName,extensionClipCullDistance:Ge&&v.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ge&&v.extensions.multiDraw===!0||qe)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:v.customProgramCacheKey()};return He.vertexUv1s=l.has(1),He.vertexUv2s=l.has(2),He.vertexUv3s=l.has(3),l.clear(),He}function A(v){let E=[];if(v.shaderID?E.push(v.shaderID):(E.push(v.customVertexShaderID),E.push(v.customFragmentShaderID)),v.defines!==void 0)for(let N in v.defines)E.push(N),E.push(v.defines[N]);return v.isRawShaderMaterial===!1&&(g(E,v),f(E,v),E.push(t.outputColorSpace)),E.push(v.customProgramCacheKey),E.join()}function g(v,E){v.push(E.precision),v.push(E.outputColorSpace),v.push(E.envMapMode),v.push(E.envMapPMREM),v.push(E.envMapMaxLod),v.push(E.envMapSize),v.push(E.mapUv),v.push(E.alphaMapUv),v.push(E.lightMapUv),v.push(E.aoMapUv),v.push(E.bumpMapUv),v.push(E.normalMapUv),v.push(E.displacementMapUv),v.push(E.emissiveMapUv),v.push(E.metalnessMapUv),v.push(E.roughnessMapUv),v.push(E.anisotropyMapUv),v.push(E.clearcoatMapUv),v.push(E.clearcoatNormalMapUv),v.push(E.clearcoatRoughnessMapUv),v.push(E.diffuseRoughnessMapUv),v.push(E.iridescenceMapUv),v.push(E.iridescenceThicknessMapUv),v.push(E.sheenColorMapUv),v.push(E.sheenRoughnessMapUv),v.push(E.specularMapUv),v.push(E.specularColorMapUv),v.push(E.specularIntensityMapUv),v.push(E.transmissionMapUv),v.push(E.thicknessMapUv),v.push(E.combine),v.push(E.fogExp2),v.push(E.sizeAttenuation),v.push(E.morphTargetsCount),v.push(E.numSunLights),v.push(E.numDirLights),v.push(E.numPointLights),v.push(E.numSpotLights),v.push(E.numSpotLightMaps),v.push(E.numHemiLights),v.push(E.numRectAreaLights),v.push(E.numSunLightShadows),v.push(E.numDirLightShadows),v.push(E.numPointLightShadows),v.push(E.numSpotLightShadows),v.push(E.numSpotLightShadowsWithMaps),v.push(E.numLightProbes),v.push(E.numLightProbeGrids),v.push(E.shadowMapType),v.push(E.toneMapping),v.push(E.numClippingPlanes),v.push(E.numClipIntersection),v.push(E.depthPacking)}function f(v,E){a.disableAll(),E.instancing&&a.enable(0),E.instancingColor&&a.enable(1),E.instancingMorph&&a.enable(2),E.matcap&&a.enable(3),E.envMap&&a.enable(4),E.normalMapObjectSpace&&a.enable(5),E.normalMapTangentSpace&&a.enable(6),E.clearcoat&&a.enable(7),E.iridescence&&a.enable(8),E.alphaTest&&a.enable(9),E.vertexColors&&a.enable(10),E.vertexAlphas&&a.enable(11),E.vertexUv1s&&a.enable(12),E.vertexUv2s&&a.enable(13),E.vertexUv3s&&a.enable(14),E.vertexTangents&&a.enable(15),E.anisotropy&&a.enable(16),E.alphaHash&&a.enable(17),E.batching&&a.enable(18),E.dispersion&&a.enable(19),E.retroreflection&&a.enable(24),E.diffuseRoughness&&a.enable(25),E.batchingColor&&a.enable(20),E.gradientMap&&a.enable(21),E.packedNormalMap&&a.enable(22),E.vertexNormals&&a.enable(23),v.push(a.mask),a.disableAll(),E.fog&&a.enable(0),E.useFog&&a.enable(1),E.flatShading&&a.enable(2),E.logarithmicDepthBuffer&&a.enable(3),E.reversedDepthBuffer&&a.enable(4),E.skinning&&a.enable(5),E.morphTargets&&a.enable(6),E.morphNormals&&a.enable(7),E.morphColors&&a.enable(8),E.premultipliedAlpha&&a.enable(9),E.shadowMapEnabled&&a.enable(10),E.doubleSided&&a.enable(11),E.flipSided&&a.enable(12),E.useDepthPacking&&a.enable(13),E.dithering&&a.enable(14),E.transmission&&a.enable(15),E.sheen&&a.enable(16),E.opaque&&a.enable(17),E.pointsUvs&&a.enable(18),E.decodeVideoTexture&&a.enable(19),E.decodeVideoTextureEmissive&&a.enable(20),E.alphaToCoverage&&a.enable(21),E.numLightProbeGrids>0&&a.enable(22),E.hasPositionAttribute&&a.enable(23),v.push(a.mask)}function m(v){let E=u[v.type],N;if(E){let C=ys[E];N=zM.clone(C.uniforms)}else N=v.uniforms;return N}function y(v,E){let N=h.get(E);return N!==void 0?++N.usedTimes:(N=new c3(t,E,v,s),c.push(N),h.set(E,N)),N}function _(v){if(--v.usedTimes===0){let E=c.indexOf(v);c[E]=c[c.length-1],c.pop(),h.delete(v.cacheKey),v.destroy()}}function T(v){o.remove(v)}function b(){o.dispose()}return{getParameters:x,getProgramCacheKey:A,getUniforms:m,acquireProgram:y,releaseProgram:_,releaseShaderCache:T,programs:c,dispose:b}}function d3(){let t=new WeakMap;function e(a){return t.has(a)}function n(a){let o=t.get(a);return o===void 0&&(o={},t.set(a,o)),o}function i(a){t.delete(a)}function s(a,o,l){t.get(a)[o]=l}function r(){t=new WeakMap}return{has:e,get:n,remove:i,update:s,dispose:r}}function p3(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.material.id!==e.material.id?t.material.id-e.material.id:t.materialVariant!==e.materialVariant?t.materialVariant-e.materialVariant:t.z!==e.z?t.z-e.z:t.id-e.id}function r1(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function a1(){let t=[],e=0,n=[],i=[],s=[];function r(){e=0,n.length=0,i.length=0,s.length=0}function a(u){let d=0;return u.isInstancedMesh&&(d+=2),u.isSkinnedMesh&&(d+=1),d}function o(u,d,x,A,g,f){let m=t[e];return m===void 0?(m={id:u.id,object:u,geometry:d,material:x,materialVariant:a(u),groupOrder:A,renderOrder:u.renderOrder,z:g,group:f},t[e]=m):(m.id=u.id,m.object=u,m.geometry=d,m.material=x,m.materialVariant=a(u),m.groupOrder=A,m.renderOrder=u.renderOrder,m.z=g,m.group=f),e++,m}function l(u,d,x,A,g,f,m){m.reversedDepth===!0&&(g=-g);let y=o(u,d,x,A,g,f);x.transmission>0?i.push(y):x.transparent===!0?s.push(y):n.push(y)}function c(u,d,x,A,g,f){let m=o(u,d,x,A,g,f);x.transmission>0?i.unshift(m):x.transparent===!0?s.unshift(m):n.unshift(m)}function h(u,d){n.length>1&&n.sort(u||p3),i.length>1&&i.sort(d||r1),s.length>1&&s.sort(d||r1)}function p(){for(let u=e,d=t.length;u<d;u++){let x=t[u];if(x.id===null)break;x.id=null,x.object=null,x.geometry=null,x.material=null,x.group=null}}return{opaque:n,transmissive:i,transparent:s,init:r,push:l,unshift:c,finish:p,sort:h}}function m3(){let t=new WeakMap;function e(i,s){let r=t.get(i),a;return r===void 0?(a=new a1,t.set(i,[a])):s>=r.length?(a=new a1,r.push(a)):a=r[s],a}function n(){t=new WeakMap}return{get:e,dispose:n}}function g3(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"SunLight":case"DirectionalLight":n={direction:new G,color:new Ye};break;case"SpotLight":n={position:new G,direction:new G,color:new Ye,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new G,color:new Ye,distance:0,decay:0};break;case"HemisphereLight":n={direction:new G,skyColor:new Ye,groundColor:new Ye};break;case"RectAreaLight":n={color:new Ye,position:new G,halfWidth:new G,halfHeight:new G};break}return t[e.id]=n,n}}}function v3(){let t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"SunLight":case"DirectionalLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Oe};break;case"SpotLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Oe};break;case"PointLight":n={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Oe,shadowCameraNear:1,shadowCameraFar:1e3};break}return t[e.id]=n,n}}}var x3=0;function y3(t,e){return(e.castShadow?2:0)-(t.castShadow?2:0)+(e.map?1:0)-(t.map?1:0)}function _3(t){let e=new g3,n=v3(),i={version:0,hash:{sunLength:-1,directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numSunShadows:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],sun:[],sunShadow:[],sunShadowMap:[],sunShadowMatrix:[],sunShadowCascade:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)i.probe.push(new G);let s=new G,r=new Ft,a=new Ft;function o(c){let h=0,p=0,u=0;for(let F=0;F<9;F++)i.probe[F].set(0,0,0);let d=0,x=0,A=0,g=0,f=0,m=0,y=0,_=0,T=0,b=0,v=0,E=0,N=0,C=0;c.sort(y3);for(let F=0,X=c.length;F<X;F++){let D=c[F],V=D.color,q=D.intensity,H=D.distance,te=null;if(D.shadow&&D.shadow.map&&(D.shadow.map.texture.format===zr?te=D.shadow.map.texture:te=D.shadow.map.depthTexture||D.shadow.map.texture),D.isAmbientLight)h+=V.r*q,p+=V.g*q,u+=V.b*q;else if(D.isLightProbe){for(let k=0;k<9;k++)i.probe[k].addScaledVector(D.sh.coefficients[k],q);C++}else if(D.isSunLight){let k=e.get(D);if(k.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){let $=D.shadow,ee=n.get(D);ee.shadowIntensity=$.intensity,ee.shadowBias=$.bias,ee.shadowNormalBias=$.normalBias,ee.shadowRadius=$.radius,ee.shadowMapSize.copy($.mapSize).multiply($.getFrameExtents()),i.sunShadow[x]=ee,i.sunShadowMap[x]=te;let _e=$.getViewportCount();for(let Ie=0;Ie<_e;Ie++)i.sunShadowMatrix[A+Ie]=$.getMatrix(Ie),i.sunShadowCascade[A+Ie]=$._cascadeData[Ie];A+=_e,x++}i.sun[d]=k,d++}else if(D.isDirectionalLight){let k=e.get(D);if(k.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){let $=D.shadow,ee=n.get(D);ee.shadowIntensity=$.intensity,ee.shadowBias=$.bias,ee.shadowNormalBias=$.normalBias,ee.shadowRadius=$.radius,ee.shadowMapSize=$.mapSize,i.directionalShadow[g]=ee,i.directionalShadowMap[g]=te,i.directionalShadowMatrix[g]=D.shadow.matrix,T++}i.directional[g]=k,g++}else if(D.isSpotLight){let k=e.get(D);k.position.setFromMatrixPosition(D.matrixWorld),k.color.copy(V).multiplyScalar(q),k.distance=H,k.coneCos=Math.cos(D.angle),k.penumbraCos=Math.cos(D.angle*(1-D.penumbra)),k.decay=D.decay,i.spot[m]=k;let $=D.shadow;if(D.map&&(i.spotLightMap[E]=D.map,E++,$.updateMatrices(D),D.castShadow&&N++),i.spotLightMatrix[m]=$.matrix,D.castShadow){let ee=n.get(D);ee.shadowIntensity=$.intensity,ee.shadowBias=$.bias,ee.shadowNormalBias=$.normalBias,ee.shadowRadius=$.radius,ee.shadowMapSize=$.mapSize,i.spotShadow[m]=ee,i.spotShadowMap[m]=te,v++}m++}else if(D.isRectAreaLight){let k=e.get(D);k.color.copy(V).multiplyScalar(q),k.halfWidth.set(D.width*.5,0,0),k.halfHeight.set(0,D.height*.5,0),i.rectArea[y]=k,y++}else if(D.isPointLight){let k=e.get(D);if(k.color.copy(D.color).multiplyScalar(D.intensity),k.distance=D.distance,k.decay=D.decay,D.castShadow){let $=D.shadow,ee=n.get(D);ee.shadowIntensity=$.intensity,ee.shadowBias=$.bias,ee.shadowNormalBias=$.normalBias,ee.shadowRadius=$.radius,ee.shadowMapSize=$.mapSize,ee.shadowCameraNear=$.camera.near,ee.shadowCameraFar=$.camera.far,i.pointShadow[f]=ee,i.pointShadowMap[f]=te,i.pointShadowMatrix[f]=D.shadow.matrix,b++}i.point[f]=k,f++}else if(D.isHemisphereLight){let k=e.get(D);k.skyColor.copy(D.color).multiplyScalar(q),k.groundColor.copy(D.groundColor).multiplyScalar(q),i.hemi[_]=k,_++}}y>0&&(t.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=ge.LTC_FLOAT_1,i.rectAreaLTC2=ge.LTC_FLOAT_2):(i.rectAreaLTC1=ge.LTC_HALF_1,i.rectAreaLTC2=ge.LTC_HALF_2)),i.ambient[0]=h,i.ambient[1]=p,i.ambient[2]=u;let B=i.hash;(B.sunLength!==d||B.directionalLength!==g||B.pointLength!==f||B.spotLength!==m||B.rectAreaLength!==y||B.hemiLength!==_||B.numSunShadows!==x||B.numDirectionalShadows!==T||B.numPointShadows!==b||B.numSpotShadows!==v||B.numSpotMaps!==E||B.numLightProbes!==C)&&(i.sun.length=d,i.directional.length=g,i.spot.length=m,i.rectArea.length=y,i.point.length=f,i.hemi.length=_,i.sunShadow.length=x,i.sunShadowMap.length=x,i.sunShadowMatrix.length=A,i.sunShadowCascade.length=A,i.directionalShadow.length=T,i.directionalShadowMap.length=T,i.directionalShadowMatrix.length=T,i.pointShadow.length=b,i.pointShadowMap.length=b,i.pointShadowMatrix.length=b,i.spotShadow.length=v,i.spotShadowMap.length=v,i.spotLightMatrix.length=v+E-N,i.spotLightMap.length=E,i.numSpotLightShadowsWithMaps=N,i.numLightProbes=C,B.sunLength=d,B.directionalLength=g,B.pointLength=f,B.spotLength=m,B.rectAreaLength=y,B.hemiLength=_,B.numSunShadows=x,B.numDirectionalShadows=T,B.numPointShadows=b,B.numSpotShadows=v,B.numSpotMaps=E,B.numLightProbes=C,i.version=x3++)}function l(c,h){let p=0,u=0,d=0,x=0,A=0,g=0,f=h.matrixWorldInverse;for(let m=0,y=c.length;m<y;m++){let _=c[m];if(_.isSunLight){let T=i.sun[p];T.direction.setFromMatrixPosition(_.matrixWorld),T.direction.transformDirection(f),p++}else if(_.isDirectionalLight){let T=i.directional[u];T.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),T.direction.sub(s),T.direction.transformDirection(f),u++}else if(_.isSpotLight){let T=i.spot[x];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(f),T.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),T.direction.sub(s),T.direction.transformDirection(f),x++}else if(_.isRectAreaLight){let T=i.rectArea[A];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(f),a.identity(),r.copy(_.matrixWorld),r.premultiply(f),a.extractRotation(r),T.halfWidth.set(_.width*.5,0,0),T.halfHeight.set(0,_.height*.5,0),T.halfWidth.applyMatrix4(a),T.halfHeight.applyMatrix4(a),A++}else if(_.isPointLight){let T=i.point[d];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(f),d++}else if(_.isHemisphereLight){let T=i.hemi[g];T.direction.setFromMatrixPosition(_.matrixWorld),T.direction.transformDirection(f),g++}}}return{setup:o,setupView:l,state:i}}function o1(t){let e=new _3(t),n=[],i=[],s=[];function r(p){h.camera=p,n.length=0,i.length=0,s.length=0}function a(p){p.isLightProbeGrid?s.push(p):n.push(p)}function o(p){i.push(p)}function l(){e.setup(n),s.sort((p,u)=>p.priority-u.priority||p.id-u.id)}function c(p){e.setupView(n,p)}let h={lightsArray:n,shadowsArray:i,lightProbeGridArray:s,camera:null,lights:e,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:h,setupLights:l,setupLightsView:c,pushLight:a,pushShadow:o}}function S3(t){let e=new WeakMap;function n(s,r=0){let a=e.get(s),o;return a===void 0?(o=new o1(t),e.set(s,[o])):r>=a.length?(o=new o1(t),a.push(o)):o=a[r],o}function i(){e=new WeakMap}return{get:n,dispose:i}}var A3=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,M3=`uniform sampler2D shadow_pass;
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
}`,E3=[new G(1,0,0),new G(-1,0,0),new G(0,1,0),new G(0,-1,0),new G(0,0,1),new G(0,0,-1)],T3=[new G(0,-1,0),new G(0,-1,0),new G(0,0,1),new G(0,0,-1),new G(0,-1,0),new G(0,-1,0)],l1=new Ft,Nc=new G,zv=new G;function b3(t,e,n){let i=new Sa,s=new Oe,r=new Oe,a=new Lt,o=new ph,l=new mh,c={},h=n.maxTextureSize,p={[ms]:Qt,[Qt]:ms,[Fn]:Fn},u=new hn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Oe},radius:{value:4}},vertexShader:A3,fragmentShader:M3}),d=u.clone();d.defines.HORIZONTAL_PASS=1;let x=new ps;x.setAttribute("position",new li(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let A=new Qn(x,u),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ec;let f=this.type;this.render=function(b,v,E){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||b.length===0)return;this.type===iM&&(De("WebGLShadowMap: PCFSoftShadowMap has been removed. Using PCFShadowMap instead."),this.type=Ec);let N=t.getRenderTarget(),C=t.getActiveCubeFace(),B=t.getActiveMipmapLevel(),F=t.state;F.setBlending(ki),F.buffers.depth.getReversed()===!0?F.buffers.color.setClear(0,0,0,0):F.buffers.color.setClear(1,1,1,1),F.buffers.depth.setTest(!0),F.setScissorTest(!1);let X=f!==this.type;X&&v.traverse(function(D){D.material&&(Array.isArray(D.material)?D.material.forEach(V=>V.needsUpdate=!0):D.material.needsUpdate=!0)});for(let D=0,V=b.length;D<V;D++){let q=b[D],H=q.shadow;if(H===void 0){De("WebGLShadowMap:",q,"has no shadow.");continue}if(H.autoUpdate===!1&&H.needsUpdate===!1)continue;s.copy(H.mapSize);let te=H.getFrameExtents();s.multiply(te),r.copy(H.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/te.x),s.x=r.x*te.x,H.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/te.y),s.y=r.y*te.y,H.mapSize.y=r.y));let k=t.state.buffers.depth.getReversed();if(H.camera._reversedDepth=k,H.map===null||X===!0){if(H.map!==null&&(H.map.depthTexture!==null&&(H.map.depthTexture.dispose(),H.map.depthTexture=null),H.map.dispose()),this.type===Zo){if(q.isPointLight){De("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}H.map=new zt(s.x,s.y,{format:zr,type:Xi,minFilter:Dt,magFilter:Dt,generateMipmaps:!1}),H.map.texture.name=q.name+".shadowMap",H.map.depthTexture=new ci(s.x,s.y,fi),H.map.depthTexture.name=q.name+".shadowMapDepth",H.map.depthTexture.format=ks,H.map.depthTexture.compareFunction=null,H.map.depthTexture.minFilter=fn,H.map.depthTexture.magFilter=fn}else q.isPointLight?(H.map=new Pc(s.x),H.map.depthTexture=new yc(s.x,Wi)):(H.map=new zt(s.x,s.y),H.map.depthTexture=new ci(s.x,s.y,Wi)),H.map.depthTexture.name=q.name+".shadowMap",H.map.depthTexture.format=ks,this.type===Ec?(H.map.depthTexture.compareFunction=k?Tv:Ev,H.map.depthTexture.minFilter=Dt,H.map.depthTexture.magFilter=Dt):(H.map.depthTexture.compareFunction=null,H.map.depthTexture.minFilter=fn,H.map.depthTexture.magFilter=fn);H.camera.updateProjectionMatrix()}H.map.isWebGLCubeRenderTarget!==!0&&(H.map.width!==s.x||H.map.height!==s.y)&&H.map.setSize(s.x,s.y);let $=H.map.isWebGLCubeRenderTarget?6:H.getViewportCount();q.isPointLight!==!0&&H.updateMatrices(q,E);let ee=(H.camera.layers.mask&4294967294)!==0?H.camera.layers:E.layers;for(let _e=0;_e<$;_e++){let Ie=H.getCamera(_e);if(q.isPointLight){let Q=H.camera,de=H.matrix,xe=q.distance||Q.far;xe!==Q.far&&(Q.far=xe,Q.updateProjectionMatrix()),Nc.setFromMatrixPosition(q.matrixWorld),Q.position.copy(Nc),zv.copy(Q.position),zv.add(E3[_e]),Q.up.copy(T3[_e]),Q.lookAt(zv),Q.updateMatrixWorld(),de.makeTranslation(-Nc.x,-Nc.y,-Nc.z),l1.multiplyMatrices(Q.projectionMatrix,Q.matrixWorldInverse),H._frustum.setFromProjectionMatrix(l1,Q.coordinateSystem,Q.reversedDepth)}if(H.map.isWebGLCubeRenderTarget)t.setRenderTarget(H.map,_e),t.clear();else{_e===0&&(t.setRenderTarget(H.map),t.clear());let Q=H.getViewport(_e);a.set(r.x*Q.x,r.y*Q.y,r.x*Q.z,r.y*Q.w),F.viewport(a)}i=H.getFrustum(_e),_(v,E,Ie,q,ee,this.type)}H.isPointLightShadow!==!0&&this.type===Zo&&m(H,E),H.needsUpdate=!1}f=this.type,g.needsUpdate=!1,t.setRenderTarget(N,C,B)};function m(b,v){let E=e.update(A);u.defines.VSM_SAMPLES!==b.blurSamples&&(u.defines.VSM_SAMPLES=b.blurSamples,d.defines.VSM_SAMPLES=b.blurSamples,u.needsUpdate=!0,d.needsUpdate=!0),b.mapPass===null?b.mapPass=new zt(s.x,s.y,{format:zr,type:Xi}):(b.mapPass.width!==b.map.width||b.mapPass.height!==b.map.height)&&b.mapPass.setSize(b.map.width,b.map.height),u.uniforms.shadow_pass.value=b.map.depthTexture,u.uniforms.resolution.value.set(b.map.width,b.map.height),u.uniforms.radius.value=b.radius,t.setRenderTarget(b.mapPass),t.clear(),t.renderBufferDirect(v,null,E,u,A,null),d.uniforms.shadow_pass.value=b.mapPass.texture,d.uniforms.resolution.value.set(b.map.width,b.map.height),d.uniforms.radius.value=b.radius,t.setRenderTarget(b.map),t.clear(),t.renderBufferDirect(v,null,E,d,A,null)}function y(b,v,E,N){let C=null,B=E.isPointLight===!0?b.customDistanceMaterial:b.customDepthMaterial;if(B!==void 0)C=B;else if(C=E.isPointLight===!0?l:o,t.localClippingEnabled&&v.clipShadows===!0&&Array.isArray(v.clippingPlanes)&&v.clippingPlanes.length!==0||v.displacementMap&&v.displacementScale!==0||v.alphaMap&&v.alphaTest>0||v.map&&v.alphaTest>0||v.alphaToCoverage===!0){let F=C.uuid,X=v.uuid,D=c[F];D===void 0&&(D={},c[F]=D);let V=D[X];V===void 0&&(V=C.clone(),D[X]=V,v.addEventListener("dispose",T)),C=V}if(C.visible=v.visible,C.wireframe=v.wireframe,N===Zo?C.side=v.shadowSide!==null?v.shadowSide:v.side:C.side=v.shadowSide!==null?v.shadowSide:p[v.side],C.alphaMap=v.alphaMap,C.alphaTest=v.alphaToCoverage===!0?.5:v.alphaTest,C.map=v.map,C.clipShadows=v.clipShadows,C.clippingPlanes=v.clippingPlanes,C.clipIntersection=v.clipIntersection,C.displacementMap=v.displacementMap,C.displacementScale=v.displacementScale,C.displacementBias=v.displacementBias,C.wireframeLinewidth=v.wireframeLinewidth,C.linewidth=v.linewidth,E.isPointLight===!0&&C.isMeshDistanceMaterial===!0){let F=t.properties.get(C);F.light=E}return C}function _(b,v,E,N,C,B){if(b.visible===!1)return;if(b.layers.test(C)&&(b.isMesh||b.isLine||b.isPoints)&&(b.castShadow||b.receiveShadow&&B===Zo)&&(!b.frustumCulled||b.intersectsFrustum(i))){b.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,b.matrixWorld);let D=e.update(b),V=b.material;if(Array.isArray(V)){let q=D.groups;for(let H=0,te=q.length;H<te;H++){let k=q[H],$=V[k.materialIndex];if($&&$.visible){let ee=y(b,$,N,B);b.onBeforeShadow(t,b,v,E,D,ee,k),t.renderBufferDirect(E,null,D,ee,b,k),b.onAfterShadow(t,b,v,E,D,ee,k)}}}else if(V.visible){let q=y(b,V,N,B);b.onBeforeShadow(t,b,v,E,D,q,null),t.renderBufferDirect(E,null,D,q,b,null),b.onAfterShadow(t,b,v,E,D,q,null)}}let X=b.children;for(let D=0,V=X.length;D<V;D++)_(X[D],v,E,N,C,B)}function T(b){b.target.removeEventListener("dispose",T);for(let E in c){let N=c[E],C=b.target.uuid;C in N&&(N[C].dispose(),delete N[C])}}}function w3(t,e){function n(){let R=!1,K=new Lt,W=null,le=new Lt(0,0,0,0);return{setMask:function(se){W!==se&&!R&&(t.colorMask(se,se,se,se),W=se)},setLocked:function(se){R=se},setClear:function(se,ae,Ee,Ge,ie){ie===!0&&(se*=Ge,ae*=Ge,Ee*=Ge),K.set(se,ae,Ee,Ge),le.equals(K)===!1&&(t.clearColor(se,ae,Ee,Ge),le.copy(K))},reset:function(){R=!1,W=null,le.set(-1,0,0,0)}}}function i(){let R=!1,K=!1,W=null,le=null,se=null;return{setReversed:function(ae){if(K!==ae){let Ee=e.get("EXT_clip_control");ae?Ee.clipControlEXT(Ee.LOWER_LEFT_EXT,Ee.ZERO_TO_ONE_EXT):Ee.clipControlEXT(Ee.LOWER_LEFT_EXT,Ee.NEGATIVE_ONE_TO_ONE_EXT),K=ae;let Ge=se;se=null,this.setClear(Ge)}},getReversed:function(){return K},setTest:function(ae){ae?pe(t.DEPTH_TEST):be(t.DEPTH_TEST)},setMask:function(ae){W!==ae&&!R&&(t.depthMask(ae),W=ae)},setFunc:function(ae){if(K&&(ae=PM[ae]),le!==ae){switch(ae){case Kf:t.depthFunc(t.NEVER);break;case Go:t.depthFunc(t.ALWAYS);break;case Jf:t.depthFunc(t.LESS);break;case Vo:t.depthFunc(t.LEQUAL);break;case jf:t.depthFunc(t.EQUAL);break;case $f:t.depthFunc(t.GEQUAL);break;case eh:t.depthFunc(t.GREATER);break;case th:t.depthFunc(t.NOTEQUAL);break;default:t.depthFunc(t.LEQUAL)}le=ae}},setLocked:function(ae){R=ae},setClear:function(ae){se!==ae&&(se=ae,K&&(ae=1-ae),t.clearDepth(ae))},reset:function(){R=!1,W=null,le=null,se=null,K=!1}}}function s(){let R=!1,K=null,W=null,le=null,se=null,ae=null,Ee=null,Ge=null,ie=null;return{setTest:function(we){R||(we?pe(t.STENCIL_TEST):be(t.STENCIL_TEST))},setMask:function(we){K!==we&&!R&&(t.stencilMask(we),K=we)},setFunc:function(we,He,me){(W!==we||le!==He||se!==me)&&(t.stencilFunc(we,He,me),W=we,le=He,se=me)},setOp:function(we,He,me){(ae!==we||Ee!==He||Ge!==me)&&(t.stencilOp(we,He,me),ae=we,Ee=He,Ge=me)},setLocked:function(we){R=we},setClear:function(we){ie!==we&&(t.clearStencil(we),ie=we)},reset:function(){R=!1,K=null,W=null,le=null,se=null,ae=null,Ee=null,Ge=null,ie=null}}}let r=new n,a=new i,o=new s,l={},c=new WeakMap,h={},p={},u={},d=new WeakMap,x=[],A=null,g=!1,f=null,m=null,y=null,_=null,T=null,b=null,v=null,E=new Ye(0,0,0),N=0,C=!1,B=null,F=null,X=null,D=null,V=null,q=t.getParameter(t.MAX_COMBINED_TEXTURE_IMAGE_UNITS),H=!1,te=0,k=t.getParameter(t.VERSION);k.indexOf("WebGL")!==-1?(te=parseFloat(/^WebGL (\d)/.exec(k)[1]),H=te>=1):k.indexOf("OpenGL ES")!==-1&&(te=parseFloat(/^OpenGL ES (\d)/.exec(k)[1]),H=te>=2);let $=null,ee={},_e=t.getParameter(t.SCISSOR_BOX),Ie=t.getParameter(t.VIEWPORT),Q=new Lt().fromArray(_e),de=new Lt().fromArray(Ie);function xe(R,K,W,le){let se=new Uint8Array(4),ae=t.createTexture();t.bindTexture(R,ae),t.texParameteri(R,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(R,t.TEXTURE_MAG_FILTER,t.NEAREST);for(let Ee=0;Ee<W;Ee++)R===t.TEXTURE_3D||R===t.TEXTURE_2D_ARRAY?t.texImage3D(K,0,t.RGBA,1,1,le,0,t.RGBA,t.UNSIGNED_BYTE,se):t.texImage2D(K+Ee,0,t.RGBA,1,1,0,t.RGBA,t.UNSIGNED_BYTE,se);return ae}let Te={};Te[t.TEXTURE_2D]=xe(t.TEXTURE_2D,t.TEXTURE_2D,1),Te[t.TEXTURE_CUBE_MAP]=xe(t.TEXTURE_CUBE_MAP,t.TEXTURE_CUBE_MAP_POSITIVE_X,6),Te[t.TEXTURE_2D_ARRAY]=xe(t.TEXTURE_2D_ARRAY,t.TEXTURE_2D_ARRAY,1,1),Te[t.TEXTURE_3D]=xe(t.TEXTURE_3D,t.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),pe(t.DEPTH_TEST),a.setFunc(Vo),zn(!1),It(tv),pe(t.CULL_FACE),dn(ki);function pe(R){h[R]!==!0&&(t.enable(R),h[R]=!0)}function be(R){h[R]!==!1&&(t.disable(R),h[R]=!1)}function qe(R,K){return u[R]!==K?(t.bindFramebuffer(R,K),u[R]=K,R===t.DRAW_FRAMEBUFFER&&(u[t.FRAMEBUFFER]=K),R===t.FRAMEBUFFER&&(u[t.DRAW_FRAMEBUFFER]=K),!0):!1}function Ce(R,K){let W=x,le=!1;if(R){W=d.get(K),W===void 0&&(W=[],d.set(K,W));let se=R.textures;if(W.length!==se.length||W[0]!==t.COLOR_ATTACHMENT0){for(let ae=0,Ee=se.length;ae<Ee;ae++)W[ae]=t.COLOR_ATTACHMENT0+ae;W.length=se.length,le=!0}}else W[0]!==t.BACK&&(W[0]=t.BACK,le=!0);le&&t.drawBuffers(W)}function st(R){return A!==R?(t.useProgram(R),A=R,!0):!1}let mt={[Ma]:t.FUNC_ADD,[rM]:t.FUNC_SUBTRACT,[aM]:t.FUNC_REVERSE_SUBTRACT};mt[oM]=t.MIN,mt[lM]=t.MAX;let Qe={[cM]:t.ZERO,[uM]:t.ONE,[fM]:t.SRC_COLOR,[rv]:t.SRC_ALPHA,[vM]:t.SRC_ALPHA_SATURATE,[mM]:t.DST_COLOR,[dM]:t.DST_ALPHA,[hM]:t.ONE_MINUS_SRC_COLOR,[av]:t.ONE_MINUS_SRC_ALPHA,[gM]:t.ONE_MINUS_DST_COLOR,[pM]:t.ONE_MINUS_DST_ALPHA,[xM]:t.CONSTANT_COLOR,[yM]:t.ONE_MINUS_CONSTANT_COLOR,[_M]:t.CONSTANT_ALPHA,[SM]:t.ONE_MINUS_CONSTANT_ALPHA};function dn(R,K,W,le,se,ae,Ee,Ge,ie,we){if(R===ki){g===!0&&(be(t.BLEND),g=!1);return}if(g===!1&&(pe(t.BLEND),g=!0),R!==sM){if(R!==f||we!==C){if((m!==Ma||T!==Ma)&&(t.blendEquation(t.FUNC_ADD),m=Ma,T=Ma),we)switch(R){case Ko:t.blendFuncSeparate(t.ONE,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case nv:t.blendFunc(t.ONE,t.ONE);break;case iv:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case sv:t.blendFuncSeparate(t.DST_COLOR,t.ONE_MINUS_SRC_ALPHA,t.ZERO,t.ONE);break;default:Be("WebGLState: Invalid blending: ",R);break}else switch(R){case Ko:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case nv:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE,t.ONE,t.ONE);break;case iv:Be("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case sv:Be("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Be("WebGLState: Invalid blending: ",R);break}y=null,_=null,b=null,v=null,E.set(0,0,0),N=0,f=R,C=we}return}se=se||K,ae=ae||W,Ee=Ee||le,(K!==m||se!==T)&&(t.blendEquationSeparate(mt[K],mt[se]),m=K,T=se),(W!==y||le!==_||ae!==b||Ee!==v)&&(t.blendFuncSeparate(Qe[W],Qe[le],Qe[ae],Qe[Ee]),y=W,_=le,b=ae,v=Ee),(Ge.equals(E)===!1||ie!==N)&&(t.blendColor(Ge.r,Ge.g,Ge.b,ie),E.copy(Ge),N=ie),f=R,C=!1}function Zt(R,K){R.side===Fn?be(t.CULL_FACE):pe(t.CULL_FACE);let W=R.side===Qt;K&&(W=!W),zn(W),R.blending===Ko&&R.transparent===!1?dn(ki):dn(R.blending,R.blendEquation,R.blendSrc,R.blendDst,R.blendEquationAlpha,R.blendSrcAlpha,R.blendDstAlpha,R.blendColor,R.blendAlpha,R.premultipliedAlpha),a.setFunc(R.depthFunc),a.setTest(R.depthTest),a.setMask(R.depthWrite),r.setMask(R.colorWrite);let le=R.stencilWrite;o.setTest(le),le&&(o.setMask(R.stencilWriteMask),o.setFunc(R.stencilFunc,R.stencilRef,R.stencilFuncMask),o.setOp(R.stencilFail,R.stencilZFail,R.stencilZPass)),nn(R.polygonOffset,R.polygonOffsetFactor,R.polygonOffsetUnits),R.alphaToCoverage===!0?pe(t.SAMPLE_ALPHA_TO_COVERAGE):be(t.SAMPLE_ALPHA_TO_COVERAGE)}function zn(R){B!==R&&(R?t.frontFace(t.CW):t.frontFace(t.CCW),B=R)}function It(R){R!==tM?(pe(t.CULL_FACE),R!==F&&(R===tv?t.cullFace(t.BACK):R===nM?t.cullFace(t.FRONT):t.cullFace(t.FRONT_AND_BACK))):be(t.CULL_FACE),F=R}function Qi(R){R!==X&&(H&&t.lineWidth(R),X=R)}function nn(R,K,W){R?(pe(t.POLYGON_OFFSET_FILL),(D!==K||V!==W)&&(D=K,V=W,a.getReversed()&&(K=-K,W=-W),t.polygonOffset(K,W))):be(t.POLYGON_OFFSET_FILL)}function di(R){R?pe(t.SCISSOR_TEST):be(t.SCISSOR_TEST)}function pi(R){R===void 0&&(R=t.TEXTURE0+q-1),$!==R&&(t.activeTexture(R),$=R)}function mi(R,K,W){W===void 0&&($===null?W=t.TEXTURE0+q-1:W=$);let le=ee[W];le===void 0&&(le={type:void 0,texture:void 0},ee[W]=le),(le.type!==R||le.texture!==K)&&($!==W&&(t.activeTexture(W),$=W),t.bindTexture(R,K||Te[R]),le.type=R,le.texture=K)}function I(){let R=ee[$];R!==void 0&&R.type!==void 0&&(t.bindTexture(R.type,null),R.type=void 0,R.texture=void 0)}function gi(){try{t.compressedTexImage2D(...arguments)}catch(R){Be("WebGLState:",R)}}function bt(){try{t.compressedTexImage3D(...arguments)}catch(R){Be("WebGLState:",R)}}function wt(){try{t.texSubImage2D(...arguments)}catch(R){Be("WebGLState:",R)}}function he(){try{t.texSubImage3D(...arguments)}catch(R){Be("WebGLState:",R)}}function Kt(){try{t.compressedTexSubImage2D(...arguments)}catch(R){Be("WebGLState:",R)}}function Re(){try{t.compressedTexSubImage3D(...arguments)}catch(R){Be("WebGLState:",R)}}function Le(){try{t.texStorage2D(...arguments)}catch(R){Be("WebGLState:",R)}}function w(){try{t.texStorage3D(...arguments)}catch(R){Be("WebGLState:",R)}}function S(){try{t.texImage2D(...arguments)}catch(R){Be("WebGLState:",R)}}function P(){try{t.texImage3D(...arguments)}catch(R){Be("WebGLState:",R)}}function ne(R){return p[R]!==void 0?p[R]:t.getParameter(R)}function j(R,K){p[R]!==K&&(t.pixelStorei(R,K),p[R]=K)}function re(R){Q.equals(R)===!1&&(t.scissor(R.x,R.y,R.z,R.w),Q.copy(R))}function ue(R){de.equals(R)===!1&&(t.viewport(R.x,R.y,R.z,R.w),de.copy(R))}function Y(R,K){let W=c.get(K);W===void 0&&(W=new WeakMap,c.set(K,W));let le=W.get(R);le===void 0&&(le=t.getUniformBlockIndex(K,R.name),t.uniformBlockBinding(K,le,le),W.set(R,le))}function Z(R,K,W){let se=c.get(K).get(R);l[se]!==W&&(t.bindBufferBase(t.UNIFORM_BUFFER,se,W),l[se]=W)}function oe(){t.disable(t.BLEND),t.disable(t.CULL_FACE),t.disable(t.DEPTH_TEST),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SCISSOR_TEST),t.disable(t.STENCIL_TEST),t.disable(t.SAMPLE_ALPHA_TO_COVERAGE),t.blendEquation(t.FUNC_ADD),t.blendFunc(t.ONE,t.ZERO),t.blendFuncSeparate(t.ONE,t.ZERO,t.ONE,t.ZERO),t.blendColor(0,0,0,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(t.LESS),a.setReversed(!1),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(t.ALWAYS,0,4294967295),t.stencilOp(t.KEEP,t.KEEP,t.KEEP),t.clearStencil(0),t.cullFace(t.BACK),t.frontFace(t.CCW),t.polygonOffset(0,0),t.activeTexture(t.TEXTURE0),t.bindFramebuffer(t.FRAMEBUFFER,null),t.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),t.bindFramebuffer(t.READ_FRAMEBUFFER,null),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),t.pixelStorei(t.PACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_ALIGNMENT,4),t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,!1),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,t.BROWSER_DEFAULT_WEBGL),t.pixelStorei(t.PACK_ROW_LENGTH,0),t.pixelStorei(t.PACK_SKIP_PIXELS,0),t.pixelStorei(t.PACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_ROW_LENGTH,0),t.pixelStorei(t.UNPACK_IMAGE_HEIGHT,0),t.pixelStorei(t.UNPACK_SKIP_PIXELS,0),t.pixelStorei(t.UNPACK_SKIP_ROWS,0),t.pixelStorei(t.UNPACK_SKIP_IMAGES,0),h={},p={},$=null,ee={},u={},l={},d=new WeakMap,x=[],A=null,g=!1,f=null,m=null,y=null,_=null,T=null,b=null,v=null,E=new Ye(0,0,0),N=0,C=!1,B=null,F=null,X=null,D=null,V=null,Q.set(0,0,t.canvas.width,t.canvas.height),de.set(0,0,t.canvas.width,t.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:pe,disable:be,bindFramebuffer:qe,drawBuffers:Ce,useProgram:st,setBlending:dn,setMaterial:Zt,setFlipSided:zn,setCullFace:It,setLineWidth:Qi,setPolygonOffset:nn,setScissorTest:di,activeTexture:pi,bindTexture:mi,unbindTexture:I,compressedTexImage2D:gi,compressedTexImage3D:bt,texImage2D:S,texImage3D:P,pixelStorei:j,getParameter:ne,updateUBOMapping:Y,uniformBlockBinding:Z,texStorage2D:Le,texStorage3D:w,texSubImage2D:wt,texSubImage3D:he,compressedTexSubImage2D:Kt,compressedTexSubImage3D:Re,scissor:re,viewport:ue,reset:oe}}function C3(t,e,n,i,s,r,a){let o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Oe,h=new WeakMap,p=new Set,u,d=new WeakMap,x=new Set,A=new FinalizationRegistry(w=>x.delete(w)),g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function f(w,S){return g?new OffscreenCanvas(w,S):dc("canvas")}function m(w,S,P){let ne=1,j=Le(w);if((j.width>P||j.height>P)&&(ne=P/Math.max(j.width,j.height)),ne<1)if(typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&w instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&w instanceof ImageBitmap||typeof VideoFrame<"u"&&w instanceof VideoFrame){let re=Math.floor(ne*j.width),ue=Math.floor(ne*j.height);u===void 0&&(u=f(re,ue));let Y=S?f(re,ue):u;return Y.width=re,Y.height=ue,Y.getContext("2d").drawImage(w,0,0,re,ue),De("WebGLRenderer: Texture has been resized from ("+j.width+"x"+j.height+") to ("+re+"x"+ue+")."),Y}else return"data"in w&&De("WebGLRenderer: Image in DataTexture is too big ("+j.width+"x"+j.height+")."),w;return w}function y(w){return w.generateMipmaps}function _(w){t.generateMipmap(w)}function T(w){return w.isWebGLCubeRenderTarget?t.TEXTURE_CUBE_MAP:w.isWebGL3DRenderTarget?t.TEXTURE_3D:w.isWebGLArrayRenderTarget||w.isCompressedArrayTexture?t.TEXTURE_2D_ARRAY:t.TEXTURE_2D}function b(w,S,P,ne,j,re=!1){if(w!==null){if(t[w]!==void 0)return t[w];De("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+w+"'")}let ue;ne&&(ue=e.get("EXT_texture_norm16"),ue||De("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let Y=S;if(S===t.RED&&(P===t.FLOAT&&(Y=t.R32F),P===t.HALF_FLOAT&&(Y=t.R16F),P===t.UNSIGNED_BYTE&&(Y=t.R8),P===t.UNSIGNED_SHORT&&ue&&(Y=ue.R16_EXT),P===t.SHORT&&ue&&(Y=ue.R16_SNORM_EXT)),S===t.RED_INTEGER&&(P===t.UNSIGNED_BYTE&&(Y=t.R8UI),P===t.UNSIGNED_SHORT&&(Y=t.R16UI),P===t.UNSIGNED_INT&&(Y=t.R32UI),P===t.BYTE&&(Y=t.R8I),P===t.SHORT&&(Y=t.R16I),P===t.INT&&(Y=t.R32I)),S===t.RG&&(P===t.FLOAT&&(Y=t.RG32F),P===t.HALF_FLOAT&&(Y=t.RG16F),P===t.UNSIGNED_BYTE&&(Y=t.RG8),P===t.UNSIGNED_SHORT&&ue&&(Y=ue.RG16_EXT),P===t.SHORT&&ue&&(Y=ue.RG16_SNORM_EXT)),S===t.RG_INTEGER&&(P===t.UNSIGNED_BYTE&&(Y=t.RG8UI),P===t.UNSIGNED_SHORT&&(Y=t.RG16UI),P===t.UNSIGNED_INT&&(Y=t.RG32UI),P===t.BYTE&&(Y=t.RG8I),P===t.SHORT&&(Y=t.RG16I),P===t.INT&&(Y=t.RG32I)),S===t.RGB_INTEGER&&(P===t.UNSIGNED_BYTE&&(Y=t.RGB8UI),P===t.UNSIGNED_SHORT&&(Y=t.RGB16UI),P===t.UNSIGNED_INT&&(Y=t.RGB32UI),P===t.BYTE&&(Y=t.RGB8I),P===t.SHORT&&(Y=t.RGB16I),P===t.INT&&(Y=t.RGB32I)),S===t.RGBA_INTEGER&&(P===t.UNSIGNED_BYTE&&(Y=t.RGBA8UI),P===t.UNSIGNED_SHORT&&(Y=t.RGBA16UI),P===t.UNSIGNED_INT&&(Y=t.RGBA32UI),P===t.BYTE&&(Y=t.RGBA8I),P===t.SHORT&&(Y=t.RGBA16I),P===t.INT&&(Y=t.RGBA32I)),S===t.RGB&&(P===t.UNSIGNED_SHORT&&ue&&(Y=ue.RGB16_EXT),P===t.SHORT&&ue&&(Y=ue.RGB16_SNORM_EXT),P===t.UNSIGNED_INT_5_9_9_9_REV&&(Y=t.RGB9_E5),P===t.UNSIGNED_INT_10F_11F_11F_REV&&(Y=t.R11F_G11F_B10F)),S===t.RGBA){let Z=re?fc:je.getTransfer(j);P===t.FLOAT&&(Y=t.RGBA32F),P===t.HALF_FLOAT&&(Y=t.RGBA16F),P===t.UNSIGNED_BYTE&&(Y=Z===ct?t.SRGB8_ALPHA8:t.RGBA8),P===t.UNSIGNED_SHORT&&ue&&(Y=ue.RGBA16_EXT),P===t.SHORT&&ue&&(Y=ue.RGBA16_SNORM_EXT),P===t.UNSIGNED_SHORT_4_4_4_4&&(Y=t.RGBA4),P===t.UNSIGNED_SHORT_5_5_5_1&&(Y=t.RGB5_A1)}return(Y===t.R16F||Y===t.R32F||Y===t.RG16F||Y===t.RG32F||Y===t.RGBA16F||Y===t.RGBA32F)&&e.get("EXT_color_buffer_float"),Y}function v(w,S){let P;return w?S===null||S===Wi||S===Ys?P=t.DEPTH24_STENCIL8:S===fi?P=t.DEPTH32F_STENCIL8:S===jo&&(P=t.DEPTH24_STENCIL8,De("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):S===null||S===Wi||S===Ys?P=t.DEPTH_COMPONENT24:S===fi?P=t.DEPTH_COMPONENT32F:S===jo&&(P=t.DEPTH_COMPONENT16),P}function E(w,S){return y(w)===!0||w.isFramebufferTexture&&w.minFilter!==fn&&w.minFilter!==Dt?Math.log2(Math.max(S.width,S.height))+1:w.mipmaps.length>0?w.mipmaps.length:w.isCompressedTexture&&Array.isArray(w.image)?S.mipmaps.length:1}function N(w){let S=new WeakRef(w);i.get(w).__ref=S,x.add(S),A.register(w,S,S)}function C(w){let S=i.get(w).__ref;S!==void 0&&(x.delete(S),A.unregister(S))}function B(w){X(w.target)}function F(w){D(w.target)}function X(w){w.removeEventListener("dispose",B),C(w),q(w),w.isVideoTexture&&h.delete(w),w.isHTMLTexture&&p.delete(w)}function D(w){w.removeEventListener("dispose",F),C(w),te(w)}function V(){for(let w of x){let S=w.deref();S!==void 0&&(S.isRenderTarget===!0?D(S):X(S))}x.clear(),p.clear()}function q(w){let S=i.get(w);if(S.__webglInit===void 0)return;let P=w.source,ne=d.get(P);if(ne){let j=ne[S.__cacheKey];j.usedTimes--,j.usedTimes===0&&H(w),Object.keys(ne).length===0&&d.delete(P)}i.remove(w)}function H(w){let S=i.get(w);t.deleteTexture(S.__webglTexture);let P=w.source,ne=d.get(P);delete ne[S.__cacheKey],a.memory.textures--}function te(w){let S=i.get(w);S.__depthDisposeCallback&&S.__depthDisposeCallback();let P=w.depthTexture;if(P&&P.renderTarget===w&&X(P),w.isWebGLCubeRenderTarget)for(let j=0;j<6;j++){if(Array.isArray(S.__webglFramebuffer[j]))for(let re=0;re<S.__webglFramebuffer[j].length;re++)t.deleteFramebuffer(S.__webglFramebuffer[j][re]);else t.deleteFramebuffer(S.__webglFramebuffer[j]);S.__webglDepthbuffer&&t.deleteRenderbuffer(S.__webglDepthbuffer[j])}else{if(Array.isArray(S.__webglFramebuffer))for(let j=0;j<S.__webglFramebuffer.length;j++)t.deleteFramebuffer(S.__webglFramebuffer[j]);else t.deleteFramebuffer(S.__webglFramebuffer);if(S.__webglDepthbuffer&&t.deleteRenderbuffer(S.__webglDepthbuffer),S.__webglMultisampledFramebuffer&&t.deleteFramebuffer(S.__webglMultisampledFramebuffer),S.__webglColorRenderbuffer)for(let j=0;j<S.__webglColorRenderbuffer.length;j++)S.__webglColorRenderbuffer[j]&&t.deleteRenderbuffer(S.__webglColorRenderbuffer[j]);S.__webglDepthRenderbuffer&&t.deleteRenderbuffer(S.__webglDepthRenderbuffer)}let ne=w.textures;for(let j=0,re=ne.length;j<re;j++){let ue=i.get(ne[j]);ue.__webglTexture&&(t.deleteTexture(ue.__webglTexture),a.memory.textures--),i.remove(ne[j])}i.remove(w)}let k=0;function $(){k=0}function ee(){return k}function _e(w){k=w}function Ie(){let w=k;return w>=s.maxTextures&&De("WebGLTextures: Trying to use "+(w+1)+" texture units while this GPU supports only "+s.maxTextures),k+=1,w}function Q(w){let S=[];return S.push(w.wrapS),S.push(w.wrapT),S.push(w.wrapR||0),S.push(w.magFilter),S.push(w.minFilter),S.push(w.anisotropy),S.push(w.internalFormat),S.push(w.format),S.push(w.type),S.push(w.generateMipmaps),S.push(w.mipmapsAutoUpdate),S.push(w.premultiplyAlpha),S.push(w.flipY),S.push(w.unpackAlignment),S.push(w.colorSpace),S.join()}function de(w,S){let P=i.get(w);if(w.isVideoTexture&&Kt(w),w.isRenderTargetTexture===!1&&w.isExternalTexture!==!0&&w.version>0&&P.__version!==w.version){let ne=w.image;if(ne===null)De("WebGLRenderer: Texture marked for update but no image data found.");else if(ne.complete===!1)De("WebGLRenderer: Texture marked for update but image is incomplete");else{Zt(P,w,S);return}}else w.isExternalTexture&&(P.__webglTexture=w.sourceTexture?w.sourceTexture:null);n.bindTexture(t.TEXTURE_2D,P.__webglTexture,t.TEXTURE0+S)}function xe(w,S){let P=i.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&P.__version!==w.version){Zt(P,w,S);return}else w.isExternalTexture&&(P.__webglTexture=w.sourceTexture?w.sourceTexture:null);n.bindTexture(t.TEXTURE_2D_ARRAY,P.__webglTexture,t.TEXTURE0+S)}function Te(w,S){let P=i.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&P.__version!==w.version){Zt(P,w,S);return}n.bindTexture(t.TEXTURE_3D,P.__webglTexture,t.TEXTURE0+S)}function pe(w,S){let P=i.get(w);if(w.version>0&&P.__version!==w.version){zn(P,w,S);return}n.bindTexture(t.TEXTURE_CUBE_MAP,P.__webglTexture,t.TEXTURE0+S)}let be={[nh]:t.REPEAT,[hs]:t.CLAMP_TO_EDGE,[ih]:t.MIRRORED_REPEAT},qe={[fn]:t.NEAREST,[EM]:t.NEAREST_MIPMAP_NEAREST,[Tc]:t.NEAREST_MIPMAP_LINEAR,[Dt]:t.LINEAR,[Dh]:t.LINEAR_MIPMAP_NEAREST,[Xs]:t.LINEAR_MIPMAP_LINEAR},Ce={[wM]:t.NEVER,[bv]:t.ALWAYS,[CM]:t.LESS,[Ev]:t.LEQUAL,[RM]:t.EQUAL,[Tv]:t.GEQUAL,[DM]:t.GREATER,[UM]:t.NOTEQUAL};function st(w,S){if(S.type===fi&&e.has("OES_texture_float_linear")===!1&&(S.magFilter===Dt||S.magFilter===Dh||S.magFilter===Tc||S.magFilter===Xs||S.minFilter===Dt||S.minFilter===Dh||S.minFilter===Tc||S.minFilter===Xs)&&De("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),t.texParameteri(w,t.TEXTURE_WRAP_S,be[S.wrapS]),t.texParameteri(w,t.TEXTURE_WRAP_T,be[S.wrapT]),(w===t.TEXTURE_3D||w===t.TEXTURE_2D_ARRAY)&&t.texParameteri(w,t.TEXTURE_WRAP_R,be[S.wrapR]),t.texParameteri(w,t.TEXTURE_MAG_FILTER,qe[S.magFilter]),t.texParameteri(w,t.TEXTURE_MIN_FILTER,qe[S.minFilter]),S.compareFunction&&(t.texParameteri(w,t.TEXTURE_COMPARE_MODE,t.COMPARE_REF_TO_TEXTURE),t.texParameteri(w,t.TEXTURE_COMPARE_FUNC,Ce[S.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(S.magFilter===fn||S.minFilter!==Tc&&S.minFilter!==Xs||S.type===fi&&e.has("OES_texture_float_linear")===!1)return;if(S.anisotropy>1||i.get(S).__currentAnisotropy){let P=e.get("EXT_texture_filter_anisotropic");t.texParameterf(w,P.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(S.anisotropy,s.getMaxAnisotropy())),i.get(S).__currentAnisotropy=S.anisotropy}}}function mt(w,S){let P=!1;w.__webglInit===void 0&&(w.__webglInit=!0,S.addEventListener("dispose",B),N(S));let ne=S.source,j=d.get(ne);j===void 0&&(j={},d.set(ne,j));let re=Q(S);if(re!==w.__cacheKey){j[re]===void 0&&(j[re]={texture:t.createTexture(),usedTimes:0},a.memory.textures++,P=!0),j[re].usedTimes++;let ue=j[w.__cacheKey];ue!==void 0&&(j[w.__cacheKey].usedTimes--,ue.usedTimes===0&&H(S)),w.__cacheKey=re,w.__webglTexture=j[re].texture}return P}function Qe(w,S,P){return Math.floor(Math.floor(w/P)/S)}function dn(w,S,P,ne){let re=w.updateRanges;if(re.length===0)n.texSubImage2D(t.TEXTURE_2D,0,0,0,S.width,S.height,P,ne,S.data);else{re.sort((R,K)=>R.start-K.start);let ue=0;for(let R=1;R<re.length;R++){let K=re[ue],W=re[R],le=K.start+K.count,se=Qe(W.start,S.width,4),ae=Qe(K.start,S.width,4);W.start<=le+1&&se===ae&&Qe(W.start+W.count-1,S.width,4)===se?K.count=Math.max(K.count,W.start+W.count-K.start):(++ue,re[ue]=W)}re.length=ue+1;let Y=n.getParameter(t.UNPACK_ROW_LENGTH),Z=n.getParameter(t.UNPACK_SKIP_PIXELS),oe=n.getParameter(t.UNPACK_SKIP_ROWS);n.pixelStorei(t.UNPACK_ROW_LENGTH,S.width);for(let R=0,K=re.length;R<K;R++){let W=re[R],le=Math.floor(W.start/4),se=Math.ceil(W.count/4),ae=le%S.width,Ee=Math.floor(le/S.width),Ge=se,ie=1;n.pixelStorei(t.UNPACK_SKIP_PIXELS,ae),n.pixelStorei(t.UNPACK_SKIP_ROWS,Ee),n.texSubImage2D(t.TEXTURE_2D,0,ae,Ee,Ge,ie,P,ne,S.data)}w.clearUpdateRanges(),n.pixelStorei(t.UNPACK_ROW_LENGTH,Y),n.pixelStorei(t.UNPACK_SKIP_PIXELS,Z),n.pixelStorei(t.UNPACK_SKIP_ROWS,oe)}}function Zt(w,S,P){let ne=t.TEXTURE_2D;(S.isDataArrayTexture||S.isCompressedArrayTexture)&&(ne=t.TEXTURE_2D_ARRAY),S.isData3DTexture&&(ne=t.TEXTURE_3D);let j=mt(w,S),re=S.source;n.bindTexture(ne,w.__webglTexture,t.TEXTURE0+P);let ue=i.get(re);if(re.version!==ue.__version||j===!0){if(n.activeTexture(t.TEXTURE0+P),(typeof ImageBitmap<"u"&&S.image instanceof ImageBitmap)===!1){let ie=je.getPrimaries(je.workingColorSpace),we=S.colorSpace===hi?null:je.getPrimaries(S.colorSpace),He=S.colorSpace===hi||ie===we?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,S.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,He)}n.pixelStorei(t.UNPACK_ALIGNMENT,S.unpackAlignment);let Z=m(S.image,!1,s.maxTextureSize);Z=Re(S,Z);let oe=r.convert(S.format,S.colorSpace),R=r.convert(S.type),K=b(S.internalFormat,oe,R,S.normalized,S.colorSpace,S.isVideoTexture);st(ne,S);let W,le=S.mipmaps,se=S.isVideoTexture!==!0,ae=ue.__version===void 0||j===!0,Ee=re.dataReady,Ge=E(S,Z);if(S.isDepthTexture)K=v(S.format===qi,S.type),ae&&(se?n.texStorage2D(t.TEXTURE_2D,1,K,Z.width,Z.height):n.texImage2D(t.TEXTURE_2D,0,K,Z.width,Z.height,0,oe,R,null));else if(S.isDataTexture)if(le.length>0){se&&ae&&n.texStorage2D(t.TEXTURE_2D,Ge,K,le[0].width,le[0].height);for(let ie=0,we=le.length;ie<we;ie++)W=le[ie],se?Ee&&n.texSubImage2D(t.TEXTURE_2D,ie,0,0,W.width,W.height,oe,R,W.data):n.texImage2D(t.TEXTURE_2D,ie,K,W.width,W.height,0,oe,R,W.data);S.generateMipmaps=!1}else se?(ae&&n.texStorage2D(t.TEXTURE_2D,Ge,K,Z.width,Z.height),Ee&&dn(S,Z,oe,R)):n.texImage2D(t.TEXTURE_2D,0,K,Z.width,Z.height,0,oe,R,Z.data);else if(S.isCompressedTexture)if(S.isCompressedArrayTexture){se&&ae&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Ge,K,le[0].width,le[0].height,Z.depth);for(let ie=0,we=le.length;ie<we;ie++)if(W=le[ie],S.format!==Yi)if(oe!==null)if(se){if(Ee)if(S.layerUpdates.size>0){let He=Iv(W.width,W.height,S.format,S.type);for(let me of S.layerUpdates){let ut=W.data.subarray(me*He/W.data.BYTES_PER_ELEMENT,(me+1)*He/W.data.BYTES_PER_ELEMENT);n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,ie,0,0,me,W.width,W.height,1,oe,ut)}}else n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,ie,0,0,0,W.width,W.height,Z.depth,oe,W.data)}else n.compressedTexImage3D(t.TEXTURE_2D_ARRAY,ie,K,W.width,W.height,Z.depth,0,W.data,0,0);else De("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else se?Ee&&n.texSubImage3D(t.TEXTURE_2D_ARRAY,ie,0,0,0,W.width,W.height,Z.depth,oe,R,W.data):n.texImage3D(t.TEXTURE_2D_ARRAY,ie,K,W.width,W.height,Z.depth,0,oe,R,W.data);S.layerUpdates.size>0&&S.clearLayerUpdates()}else{se&&ae&&n.texStorage2D(t.TEXTURE_2D,Ge,K,le[0].width,le[0].height);for(let ie=0,we=le.length;ie<we;ie++)W=le[ie],S.format!==Yi?oe!==null?se?Ee&&n.compressedTexSubImage2D(t.TEXTURE_2D,ie,0,0,W.width,W.height,oe,W.data):n.compressedTexImage2D(t.TEXTURE_2D,ie,K,W.width,W.height,0,W.data):De("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):se?Ee&&n.texSubImage2D(t.TEXTURE_2D,ie,0,0,W.width,W.height,oe,R,W.data):n.texImage2D(t.TEXTURE_2D,ie,K,W.width,W.height,0,oe,R,W.data)}else if(S.isDataArrayTexture)if(se){if(ae&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Ge,K,Z.width,Z.height,Z.depth),Ee)if(S.layerUpdates.size>0){let ie=Iv(Z.width,Z.height,S.format,S.type);for(let we of S.layerUpdates){let He=Z.data.subarray(we*ie/Z.data.BYTES_PER_ELEMENT,(we+1)*ie/Z.data.BYTES_PER_ELEMENT);n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,we,Z.width,Z.height,1,oe,R,He)}S.clearLayerUpdates()}else n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,0,Z.width,Z.height,Z.depth,oe,R,Z.data)}else n.texImage3D(t.TEXTURE_2D_ARRAY,0,K,Z.width,Z.height,Z.depth,0,oe,R,Z.data);else if(S.isData3DTexture)se?(ae&&n.texStorage3D(t.TEXTURE_3D,Ge,K,Z.width,Z.height,Z.depth),Ee&&n.texSubImage3D(t.TEXTURE_3D,0,0,0,0,Z.width,Z.height,Z.depth,oe,R,Z.data)):n.texImage3D(t.TEXTURE_3D,0,K,Z.width,Z.height,Z.depth,0,oe,R,Z.data);else if(S.isFramebufferTexture){if(ae)if(se)n.texStorage2D(t.TEXTURE_2D,Ge,K,Z.width,Z.height);else{let ie=Z.width,we=Z.height;for(let He=0;He<Ge;He++)n.texImage2D(t.TEXTURE_2D,He,K,ie,we,0,oe,R,null),ie>>=1,we>>=1}}else if(S.isHTMLTexture){if("texElementImage2D"in t){let ie=t.canvas;if(ie.hasAttribute("layoutsubtree")||ie.setAttribute("layoutsubtree","true"),Z.parentNode!==ie){ie.appendChild(Z),p.add(S),ie.onpaint=we=>{let He=we.changedElements;for(let me of p)He.includes(me.image)&&(me.needsUpdate=!0)},ie.requestPaint();return}if(t.texElementImage2D.length===3)t.texElementImage2D(t.TEXTURE_2D,t.RGBA8,Z);else{let He=t.RGBA,me=t.RGBA,ut=t.UNSIGNED_BYTE;t.texElementImage2D(t.TEXTURE_2D,0,He,me,ut,Z)}t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,t.LINEAR),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE)}}else if(le.length>0){if(se&&ae){let ie=Le(le[0]);n.texStorage2D(t.TEXTURE_2D,Ge,K,ie.width,ie.height)}for(let ie=0,we=le.length;ie<we;ie++)W=le[ie],se?Ee&&n.texSubImage2D(t.TEXTURE_2D,ie,0,0,oe,R,W):n.texImage2D(t.TEXTURE_2D,ie,K,oe,R,W);S.generateMipmaps=!1}else if(se){if(ae){let ie=Le(Z);n.texStorage2D(t.TEXTURE_2D,Ge,K,ie.width,ie.height)}Ee&&n.texSubImage2D(t.TEXTURE_2D,0,0,0,oe,R,Z)}else n.texImage2D(t.TEXTURE_2D,0,K,oe,R,Z);y(S)&&S.mipmapsAutoUpdate===!0&&_(ne),ue.__version=re.version,S.onUpdate&&S.onUpdate(S)}w.__version=S.version}function zn(w,S,P){if(S.image.length!==6)return;let ne=mt(w,S),j=S.source;n.bindTexture(t.TEXTURE_CUBE_MAP,w.__webglTexture,t.TEXTURE0+P);let re=i.get(j);if(j.version!==re.__version||ne===!0){n.activeTexture(t.TEXTURE0+P);let ue=je.getPrimaries(je.workingColorSpace),Y=S.colorSpace===hi?null:je.getPrimaries(S.colorSpace),Z=S.colorSpace===hi||ue===Y?t.NONE:t.BROWSER_DEFAULT_WEBGL;n.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,S.flipY),n.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),n.pixelStorei(t.UNPACK_ALIGNMENT,S.unpackAlignment),n.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,Z);let oe=S.isCompressedTexture||S.image[0].isCompressedTexture,R=S.image[0]&&S.image[0].isDataTexture,K=[];for(let me=0;me<6;me++)!oe&&!R?K[me]=m(S.image[me],!0,s.maxCubemapSize):K[me]=R?S.image[me].image:S.image[me],K[me]=Re(S,K[me]);let W=K[0],le=r.convert(S.format,S.colorSpace),se=r.convert(S.type),ae=b(S.internalFormat,le,se,S.normalized,S.colorSpace),Ee=S.isVideoTexture!==!0,Ge=re.__version===void 0||ne===!0,ie=j.dataReady,we=E(S,W);st(t.TEXTURE_CUBE_MAP,S);let He;if(S.isDepthTexture){if(ae=v(S.format===qi,S.type),Ge)if(Ee)n.texStorage2D(t.TEXTURE_CUBE_MAP,1,ae,W.width,W.height);else for(let me=0;me<6;me++)n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,0,ae,W.width,W.height,0,le,se,null)}else if(oe){Ee&&Ge&&n.texStorage2D(t.TEXTURE_CUBE_MAP,we,ae,W.width,W.height);for(let me=0;me<6;me++){He=K[me].mipmaps;for(let ut=0;ut<He.length;ut++){let sn=He[ut];S.format!==Yi?le!==null?Ee?ie&&n.compressedTexSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut,0,0,sn.width,sn.height,le,sn.data):n.compressedTexImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut,ae,sn.width,sn.height,0,sn.data):De("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):Ee?ie&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut,0,0,sn.width,sn.height,le,se,sn.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut,ae,sn.width,sn.height,0,le,se,sn.data)}}}else{if(He=S.mipmaps,Ee&&Ge){He.length>0&&we++;let me=Le(K[0]);n.texStorage2D(t.TEXTURE_CUBE_MAP,we,ae,me.width,me.height)}for(let me=0;me<6;me++)if(R){Ee?ie&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,0,0,0,K[me].width,K[me].height,le,se,K[me].data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,0,ae,K[me].width,K[me].height,0,le,se,K[me].data);for(let ut=0;ut<He.length;ut++){let qs=He[ut].image[me].image;Ee?ie&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut+1,0,0,qs.width,qs.height,le,se,qs.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut+1,ae,qs.width,qs.height,0,le,se,qs.data)}}else{Ee?ie&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,0,0,0,le,se,K[me]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,0,ae,le,se,K[me]);for(let ut=0;ut<He.length;ut++){let sn=He[ut];Ee?ie&&n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut+1,0,0,le,se,sn.image[me]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+me,ut+1,ae,le,se,sn.image[me])}}}y(S)&&S.mipmapsAutoUpdate===!0&&_(t.TEXTURE_CUBE_MAP),re.__version=j.version,S.onUpdate&&S.onUpdate(S)}w.__version=S.version}function It(w,S,P,ne,j,re){let ue=r.convert(P.format,P.colorSpace),Y=r.convert(P.type),Z=b(P.internalFormat,ue,Y,P.normalized,P.colorSpace),oe=i.get(S),R=i.get(P);if(R.__renderTarget=S,!oe.__hasExternalTextures){let K=Math.max(1,S.width>>re),W=Math.max(1,S.height>>re);j===t.TEXTURE_3D||j===t.TEXTURE_2D_ARRAY?n.texImage3D(j,re,Z,K,W,S.depth,0,ue,Y,null):n.texImage2D(j,re,Z,K,W,0,ue,Y,null)}n.bindFramebuffer(t.FRAMEBUFFER,w),he(S)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,ne,j,R.__webglTexture,0,wt(S)):(j===t.TEXTURE_2D||j>=t.TEXTURE_CUBE_MAP_POSITIVE_X&&j<=t.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&t.framebufferTexture2D(t.FRAMEBUFFER,ne,j,R.__webglTexture,re),n.bindFramebuffer(t.FRAMEBUFFER,null)}function Qi(w,S,P){if(t.bindRenderbuffer(t.RENDERBUFFER,w),S.depthBuffer){let ne=S.depthTexture,j=ne&&ne.isDepthTexture?ne.type:null,re=v(S.stencilBuffer,j),ue=S.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;he(S)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,wt(S),re,S.width,S.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,wt(S),re,S.width,S.height):t.renderbufferStorage(t.RENDERBUFFER,re,S.width,S.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,ue,t.RENDERBUFFER,w)}else{let ne=S.textures;for(let j=0;j<ne.length;j++){let re=ne[j],ue=r.convert(re.format,re.colorSpace),Y=r.convert(re.type),Z=b(re.internalFormat,ue,Y,re.normalized,re.colorSpace);he(S)?o.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,wt(S),Z,S.width,S.height):P?t.renderbufferStorageMultisample(t.RENDERBUFFER,wt(S),Z,S.width,S.height):t.renderbufferStorage(t.RENDERBUFFER,Z,S.width,S.height)}}t.bindRenderbuffer(t.RENDERBUFFER,null)}function nn(w,S,P){let ne=S.isWebGLCubeRenderTarget===!0;if(n.bindFramebuffer(t.FRAMEBUFFER,w),!(S.depthTexture&&S.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let j=i.get(S.depthTexture);j.__renderTarget=S;let re=ne?S.depthTexture.image[0]:S.depthTexture.image;(!j.__webglTexture||re.width!==S.width||re.height!==S.height)&&(re.width=S.width,re.height=S.height,S.depthTexture.needsUpdate=!0),ne?pe(S.depthTexture,0):de(S.depthTexture,0);let ue=j.__webglTexture,Y=wt(S),Z=ne?t.TEXTURE_CUBE_MAP_POSITIVE_X+P:t.TEXTURE_2D,oe=S.depthTexture.format===qi?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;if(S.depthTexture.format===ks)he(S)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,oe,Z,ue,0,Y):t.framebufferTexture2D(t.FRAMEBUFFER,oe,Z,ue,0);else if(S.depthTexture.format===qi)he(S)?o.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,oe,Z,ue,0,Y):t.framebufferTexture2D(t.FRAMEBUFFER,oe,Z,ue,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function di(w){let S=i.get(w),P=w.isWebGLCubeRenderTarget===!0;if(S.__boundDepthTexture!==w.depthTexture){let ne=w.depthTexture;if(S.__depthDisposeCallback&&S.__depthDisposeCallback(),ne){let j=()=>{delete S.__boundDepthTexture,delete S.__depthDisposeCallback,ne.removeEventListener("dispose",j)};ne.addEventListener("dispose",j),S.__depthDisposeCallback=j}S.__boundDepthTexture=ne}if(w.depthTexture&&!S.__autoAllocateDepthBuffer)if(P)for(let ne=0;ne<6;ne++)nn(S.__webglFramebuffer[ne],w,ne);else w.texture.mipmaps.length>0?nn(S.__webglFramebuffer[0],w,0):nn(S.__webglFramebuffer,w,0);else if(P){S.__webglDepthbuffer=[];for(let ne=0;ne<6;ne++)if(n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer[ne]),S.__webglDepthbuffer[ne]===void 0)S.__webglDepthbuffer[ne]=t.createRenderbuffer(),Qi(S.__webglDepthbuffer[ne],w,!1);else{let j=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,re=S.__webglDepthbuffer[ne];t.bindRenderbuffer(t.RENDERBUFFER,re),t.framebufferRenderbuffer(t.FRAMEBUFFER,j,t.RENDERBUFFER,re)}}else if(w.texture.mipmaps.length>0?n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer[0]):n.bindFramebuffer(t.FRAMEBUFFER,S.__webglFramebuffer),S.__webglDepthbuffer===void 0)S.__webglDepthbuffer=t.createRenderbuffer(),Qi(S.__webglDepthbuffer,w,!1);else{let j=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,re=S.__webglDepthbuffer;t.bindRenderbuffer(t.RENDERBUFFER,re),t.framebufferRenderbuffer(t.FRAMEBUFFER,j,t.RENDERBUFFER,re)}n.bindFramebuffer(t.FRAMEBUFFER,null)}function pi(w,S,P){let ne=i.get(w);S!==void 0&&It(ne.__webglFramebuffer,w,w.texture,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,0),P!==void 0&&di(w)}function mi(w){let S=w.texture,P=i.get(w),ne=i.get(S);w.addEventListener("dispose",F),N(w);let j=w.textures,re=w.isWebGLCubeRenderTarget===!0,ue=j.length>1;if(ue||(ne.__webglTexture===void 0&&(ne.__webglTexture=t.createTexture()),ne.__version=S.version,a.memory.textures++),re){P.__webglFramebuffer=[];for(let Y=0;Y<6;Y++)if(S.mipmaps.length>0){P.__webglFramebuffer[Y]=[];for(let Z=0;Z<S.mipmaps.length;Z++)P.__webglFramebuffer[Y][Z]=t.createFramebuffer()}else P.__webglFramebuffer[Y]=t.createFramebuffer()}else{if(S.mipmaps.length>0){P.__webglFramebuffer=[];for(let Y=0;Y<S.mipmaps.length;Y++)P.__webglFramebuffer[Y]=t.createFramebuffer()}else P.__webglFramebuffer=t.createFramebuffer();if(ue)for(let Y=0,Z=j.length;Y<Z;Y++){let oe=i.get(j[Y]);oe.__webglTexture===void 0&&(oe.__webglTexture=t.createTexture(),a.memory.textures++)}if(w.samples>0&&he(w)===!1){P.__webglMultisampledFramebuffer=t.createFramebuffer(),P.__webglColorRenderbuffer=[],n.bindFramebuffer(t.FRAMEBUFFER,P.__webglMultisampledFramebuffer);for(let Y=0;Y<j.length;Y++){let Z=j[Y];P.__webglColorRenderbuffer[Y]=t.createRenderbuffer(),t.bindRenderbuffer(t.RENDERBUFFER,P.__webglColorRenderbuffer[Y]);let oe=r.convert(Z.format,Z.colorSpace),R=r.convert(Z.type),K=b(Z.internalFormat,oe,R,Z.normalized,Z.colorSpace,w.isXRRenderTarget===!0),W=wt(w);t.renderbufferStorageMultisample(t.RENDERBUFFER,W,K,w.width,w.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+Y,t.RENDERBUFFER,P.__webglColorRenderbuffer[Y])}t.bindRenderbuffer(t.RENDERBUFFER,null),w.depthBuffer&&(P.__webglDepthRenderbuffer=t.createRenderbuffer(),Qi(P.__webglDepthRenderbuffer,w,!0)),n.bindFramebuffer(t.FRAMEBUFFER,null)}}if(re){n.bindTexture(t.TEXTURE_CUBE_MAP,ne.__webglTexture),st(t.TEXTURE_CUBE_MAP,S);for(let Y=0;Y<6;Y++)if(S.mipmaps.length>0)for(let Z=0;Z<S.mipmaps.length;Z++)It(P.__webglFramebuffer[Y][Z],w,S,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+Y,Z);else It(P.__webglFramebuffer[Y],w,S,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+Y,0);y(S)?_(t.TEXTURE_CUBE_MAP):S.mipmaps.length>0&&t.texParameteri(t.TEXTURE_CUBE_MAP,t.TEXTURE_MAX_LEVEL,S.mipmaps.length-1),n.unbindTexture()}else if(ue){for(let Y=0,Z=j.length;Y<Z;Y++){let oe=j[Y],R=i.get(oe),K=t.TEXTURE_2D;(w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(K=w.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(K,R.__webglTexture),st(K,oe),It(P.__webglFramebuffer,w,oe,t.COLOR_ATTACHMENT0+Y,K,0),y(oe)&&_(K)}n.unbindTexture()}else{let Y=t.TEXTURE_2D;if((w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(Y=w.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY),n.bindTexture(Y,ne.__webglTexture),st(Y,S),S.mipmaps.length>0)for(let Z=0;Z<S.mipmaps.length;Z++)It(P.__webglFramebuffer[Z],w,S,t.COLOR_ATTACHMENT0,Y,Z);else It(P.__webglFramebuffer,w,S,t.COLOR_ATTACHMENT0,Y,0);y(S)?_(Y):S.mipmaps.length>0&&t.texParameteri(Y,t.TEXTURE_MAX_LEVEL,S.mipmaps.length-1),n.unbindTexture()}w.depthBuffer&&di(w)}function I(w){let S=w.textures;for(let P=0,ne=S.length;P<ne;P++){let j=S[P];if(y(j)&&j.mipmapsAutoUpdate===!0){let re=T(w),ue=i.get(j).__webglTexture;n.bindTexture(re,ue),_(re),n.unbindTexture()}}}let gi=[];function bt(w){if(w.samples>0){if(he(w)===!1){let S=w.textures,P=w.width,ne=w.height,j=t.COLOR_BUFFER_BIT,re=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,ue=i.get(w),Y=S.length>1;if(Y)for(let oe=0;oe<S.length;oe++)n.bindFramebuffer(t.FRAMEBUFFER,ue.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.RENDERBUFFER,null),n.bindFramebuffer(t.FRAMEBUFFER,ue.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.TEXTURE_2D,null,0);n.bindFramebuffer(t.READ_FRAMEBUFFER,ue.__webglMultisampledFramebuffer),w.texture.mipmaps.length>0?n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ue.__webglFramebuffer[0]):n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ue.__webglFramebuffer);for(let oe=0;oe<S.length;oe++){if(w.resolveDepthBuffer&&(w.depthBuffer&&(j|=t.DEPTH_BUFFER_BIT),w.stencilBuffer&&w.resolveStencilBuffer&&(j|=t.STENCIL_BUFFER_BIT)),Y){t.framebufferRenderbuffer(t.READ_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.RENDERBUFFER,ue.__webglColorRenderbuffer[oe]);let R=i.get(S[oe]).__webglTexture;t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,R,0)}t.blitFramebuffer(0,0,P,ne,0,0,P,ne,j,t.NEAREST),l===!0&&(gi.length=0,gi.push(t.COLOR_ATTACHMENT0+oe),w.depthBuffer&&w.storeMultisampledDepthBuffer===!1&&gi.push(re),t.invalidateFramebuffer(t.READ_FRAMEBUFFER,gi))}if(n.bindFramebuffer(t.READ_FRAMEBUFFER,null),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),Y)for(let oe=0;oe<S.length;oe++){n.bindFramebuffer(t.FRAMEBUFFER,ue.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.RENDERBUFFER,ue.__webglColorRenderbuffer[oe]);let R=i.get(S[oe]).__webglTexture;n.bindFramebuffer(t.FRAMEBUFFER,ue.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+oe,t.TEXTURE_2D,R,0)}n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ue.__webglMultisampledFramebuffer)}else if(w.depthBuffer&&w.storeMultisampledDepthBuffer===!1&&l){let S=w.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT;t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,[S])}}}function wt(w){return Math.min(s.maxSamples,w.samples)}function he(w){let S=i.get(w);return w.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&S.__useRenderToTexture!==!1}function Kt(w){let S=a.render.frame;h.get(w)!==S&&(h.set(w,S),w.update())}function Re(w,S){let P=w.colorSpace,ne=w.format,j=w.type;return w.isCompressedTexture===!0||w.isVideoTexture===!0||P!==Ws&&P!==hi&&(je.getTransfer(P)===ct?(ne!==Yi||j!==Gt)&&De("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Be("WebGLTextures: Unsupported texture color space:",P)),S}function Le(w){return typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement?(c.width=w.naturalWidth||w.width,c.height=w.naturalHeight||w.height):typeof VideoFrame<"u"&&w instanceof VideoFrame?(c.width=w.displayWidth,c.height=w.displayHeight):(c.width=w.width,c.height=w.height),c}this.allocateTextureUnit=Ie,this.resetTextureUnits=$,this.getTextureUnits=ee,this.setTextureUnits=_e,this.setTexture2D=de,this.setTexture2DArray=xe,this.setTexture3D=Te,this.setTextureCube=pe,this.rebindTextures=pi,this.setupRenderTarget=mi,this.updateRenderTargetMipmap=I,this.updateMultisampleRenderTarget=bt,this.setupDepthRenderbuffer=di,this.setupFrameBufferTexture=It,this.useMultisampledRTT=he,this.dispose=V}function R3(t,e){function n(i,s=hi){let r,a=je.getTransfer(s);if(i===Gt)return t.UNSIGNED_BYTE;if(i===Ih)return t.UNSIGNED_SHORT_4_4_4_4;if(i===Bh)return t.UNSIGNED_SHORT_5_5_5_1;if(i===xv)return t.UNSIGNED_INT_5_9_9_9_REV;if(i===yv)return t.UNSIGNED_INT_10F_11F_11F_REV;if(i===gv)return t.BYTE;if(i===vv)return t.SHORT;if(i===jo)return t.UNSIGNED_SHORT;if(i===Uh)return t.INT;if(i===Wi)return t.UNSIGNED_INT;if(i===fi)return t.FLOAT;if(i===Xi)return t.HALF_FLOAT;if(i===_v)return t.ALPHA;if(i===Sv)return t.RGB;if(i===Yi)return t.RGBA;if(i===ks)return t.DEPTH_COMPONENT;if(i===qi)return t.DEPTH_STENCIL;if(i===Av)return t.RED;if(i===Nh)return t.RED_INTEGER;if(i===zr)return t.RG;if(i===Lh)return t.RG_INTEGER;if(i===Ph)return t.RGBA_INTEGER;if(i===bc||i===wc||i===Cc||i===Rc)if(a===ct)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===bc)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===wc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===Cc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===Rc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===bc)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===wc)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===Cc)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===Rc)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===Oh||i===Fh||i===zh||i===Hh)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===Oh)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===Fh)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===zh)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===Hh)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===Gh||i===Vh||i===kh||i===Wh||i===Xh||i===Dc||i===Yh)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===Gh||i===Vh)return a===ct?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===kh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(i===Wh)return r.COMPRESSED_R11_EAC;if(i===Xh)return r.COMPRESSED_SIGNED_R11_EAC;if(i===Dc)return r.COMPRESSED_RG11_EAC;if(i===Yh)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(i===qh||i===Qh||i===Zh||i===Kh||i===Jh||i===jh||i===$h||i===ed||i===td||i===nd||i===id||i===sd||i===rd||i===ad)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===qh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===Qh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===Zh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===Kh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===Jh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===jh)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===$h)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===ed)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===td)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===nd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===id)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===sd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===rd)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===ad)return a===ct?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===od||i===ld||i===cd)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===od)return a===ct?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===ld)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===cd)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===ud||i===fd||i===Uc||i===hd)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===ud)return r.COMPRESSED_RED_RGTC1_EXT;if(i===fd)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===Uc)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===hd)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===Ys?t.UNSIGNED_INT_24_8:t[i]!==void 0?t[i]:null}return{convert:n}}var D3=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,U3=`
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

}`,Yv=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,n){if(this.texture===null){let i=new _c(e.texture);(e.depthNear!==n.depthNear||e.depthFar!==n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){let n=e.cameras[0].viewport,i=new hn({vertexShader:D3,fragmentShader:U3,uniforms:{depthColor:{value:this.texture},depthWidth:{value:n.z},depthHeight:{value:n.w}}});this.mesh=new Qn(new Aa(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},qv=class extends qn{constructor(e,n){super();let i=this,s=null,r=1,a=null,o="local-floor",l=1,c=null,h=null,p=null,u=null,d=null,x=null,A=typeof XRWebGLBinding<"u",g=new Yv,f={},m=n.getContextAttributes(),y=null,_=null,T=[],b=[],v=new Oe,E=null,N=null,C=new On;C.viewport=new Lt;let B=new On;B.viewport=new Lt;let F=[C,B],X=new bh,D=null,V=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(Q){let de=T[Q];return de===void 0&&(de=new Xo,T[Q]=de),de.getTargetRaySpace()},this.getControllerGrip=function(Q){let de=T[Q];return de===void 0&&(de=new Xo,T[Q]=de),de.getGripSpace()},this.getHand=function(Q){let de=T[Q];return de===void 0&&(de=new Xo,T[Q]=de),de.getHandSpace()};function q(Q){let de=b.indexOf(Q.inputSource);if(de===-1)return;let xe=T[de];xe!==void 0&&(xe.update(Q.inputSource,Q.frame,c||a),xe.dispatchEvent({type:Q.type,data:Q.inputSource}))}function H(){s.removeEventListener("select",q),s.removeEventListener("selectstart",q),s.removeEventListener("selectend",q),s.removeEventListener("squeeze",q),s.removeEventListener("squeezestart",q),s.removeEventListener("squeezeend",q),s.removeEventListener("end",H),s.removeEventListener("inputsourceschange",te);for(let Q=0;Q<T.length;Q++){let de=b[Q];de!==null&&(b[Q]=null,T[Q].disconnect(de))}D=null,V=null,g.reset();for(let Q in f)delete f[Q];if(e.setRenderTarget(y),d=null,u=null,p=null,s=null,_=null,Ie.stop(),i.isPresenting=!1,e.setPixelRatio(E),e.setSize(v.width,v.height,!1),N!==null){let Q=N.camera;Q.fov=N.fov,Q.zoom=N.zoom,Q.updateProjectionMatrix(),N=null}i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(Q){r=Q,i.isPresenting===!0&&De("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(Q){o=Q,i.isPresenting===!0&&De("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||a},this.setReferenceSpace=function(Q){c=Q},this.getBaseLayer=function(){return u!==null?u:d},this.getBinding=function(){return p===null&&A&&(p=new XRWebGLBinding(s,n)),p},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(Q){if(s=Q,s!==null){if(y=e.getRenderTarget(),s.addEventListener("select",q),s.addEventListener("selectstart",q),s.addEventListener("selectend",q),s.addEventListener("squeeze",q),s.addEventListener("squeezestart",q),s.addEventListener("squeezeend",q),s.addEventListener("end",H),s.addEventListener("inputsourceschange",te),m.xrCompatible!==!0&&await n.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(v),A&&"createProjectionLayer"in XRWebGLBinding.prototype){let xe=null,Te=null,pe=null;m.depth&&(pe=m.stencil?n.DEPTH24_STENCIL8:n.DEPTH_COMPONENT24,xe=m.stencil?qi:ks,Te=m.stencil?Ys:Wi);let be={colorFormat:n.RGBA8,depthFormat:pe,scaleFactor:r};p=this.getBinding(),u=p.createProjectionLayer(be),s.updateRenderState({layers:[u]}),e.setPixelRatio(1),e.setSize(u.textureWidth,u.textureHeight,!1),_=new zt(u.textureWidth,u.textureHeight,{format:Yi,type:Gt,depthTexture:new ci(u.textureWidth,u.textureHeight,Te,void 0,void 0,void 0,void 0,void 0,void 0,xe),stencilBuffer:m.stencil,colorSpace:e.outputColorSpace,samples:m.antialias?4:0,resolveDepthBuffer:u.ignoreDepthValues===!1,resolveStencilBuffer:u.ignoreDepthValues===!1,storeMultisampledDepthBuffer:u.ignoreDepthValues===!1,storeMultisampledStencilBuffer:u.ignoreDepthValues===!1})}else{let xe={antialias:m.antialias,alpha:!0,depth:m.depth,stencil:m.stencil,framebufferScaleFactor:r};d=new XRWebGLLayer(s,n,xe),s.updateRenderState({baseLayer:d}),e.setPixelRatio(1),e.setSize(d.framebufferWidth,d.framebufferHeight,!1),_=new zt(d.framebufferWidth,d.framebufferHeight,{format:Yi,type:Gt,colorSpace:e.outputColorSpace,stencilBuffer:m.stencil,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1,storeMultisampledDepthBuffer:d.ignoreDepthValues===!1,storeMultisampledStencilBuffer:d.ignoreDepthValues===!1})}_.isXRRenderTarget=!0,this.setFoveation(l),c=null,a=await s.requestReferenceSpace(o),Ie.setContext(s),Ie.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function te(Q){for(let de=0;de<Q.removed.length;de++){let xe=Q.removed[de],Te=b.indexOf(xe);Te>=0&&(b[Te]=null,T[Te].disconnect(xe))}for(let de=0;de<Q.added.length;de++){let xe=Q.added[de],Te=b.indexOf(xe);if(Te===-1){for(let be=0;be<T.length;be++)if(be>=b.length){b.push(xe),Te=be;break}else if(b[be]===null){b[be]=xe,Te=be;break}if(Te===-1)break}let pe=T[Te];pe&&pe.connect(xe)}}function k(Q,de){de===null?Q.matrixWorld.copy(Q.matrix):Q.matrixWorld.multiplyMatrices(de.matrixWorld,Q.matrix),Q.matrixWorldInverse.copy(Q.matrixWorld).invert()}this.updateCamera=function(Q){if(s===null)return;let de=Q.near,xe=Q.far;g.texture!==null&&(g.depthNear>0&&(de=g.depthNear),g.depthFar>0&&(xe=g.depthFar)),X.near=B.near=C.near=de,X.far=B.far=C.far=xe,(D!==X.near||V!==X.far)&&(s.updateRenderState({depthNear:X.near,depthFar:X.far}),D=X.near,V=X.far),X.layers.mask=Q.layers.mask|6,C.layers.mask=X.layers.mask&-5,B.layers.mask=X.layers.mask&-3;let Te=Q.parent,pe=X.cameras;k(X,Te);for(let be=0;be<pe.length;be++)k(pe[be],Te);X.projectionMatrix.copy(C.projectionMatrix),X.projectionMatrixInverse.copy(C.projectionMatrixInverse),N===null&&Q.isPerspectiveCamera&&(N={camera:Q,fov:Q.fov,zoom:Q.zoom}),$(Q,X,Te)};function $(Q,de,xe){xe===null?Q.matrix.copy(de.matrixWorld):(Q.matrix.copy(xe.matrixWorld),Q.matrix.invert(),Q.matrix.multiply(de.matrixWorld)),Q.matrix.decompose(Q.position,Q.quaternion,Q.scale),Q.updateMatrixWorld(!0),Q.projectionMatrix.copy(de.projectionMatrix),Q.projectionMatrixInverse.copy(de.projectionMatrixInverse),Q.isPerspectiveCamera&&(Q.fov=rh*2*Math.atan(1/Q.projectionMatrix.elements[5]),Q.zoom=1)}this.getCamera=function(){return X},this.getFoveation=function(){if(!(u===null&&d===null))return l},this.setFoveation=function(Q){l=Q,u!==null&&(u.fixedFoveation=Q),d!==null&&d.fixedFoveation!==void 0&&(d.fixedFoveation=Q)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(X)},this.getCameraTexture=function(Q){return f[Q]};let ee=null;function _e(Q,de){if(h=de.getViewerPose(c||a),x=de,h!==null){let xe=h.views;d!==null&&(e.setRenderTargetFramebuffer(_,d.framebuffer),e.setRenderTarget(_));let Te=!1;xe.length!==X.cameras.length&&(X.cameras.length=0,Te=!0);for(let Ce=0;Ce<xe.length;Ce++){let st=xe[Ce],mt=null;if(d!==null)mt=d.getViewport(st);else{let dn=p.getViewSubImage(u,st);mt=dn.viewport,Ce===0&&(e.setRenderTargetTextures(_,dn.colorTexture,dn.depthStencilTexture),e.setRenderTarget(_))}let Qe=F[Ce];Qe===void 0&&(Qe=new On,Qe.layers.enable(Ce),Qe.viewport=new Lt,F[Ce]=Qe),Qe.matrix.fromArray(st.transform.matrix),Qe.matrix.decompose(Qe.position,Qe.quaternion,Qe.scale),Qe.projectionMatrix.fromArray(st.projectionMatrix),Qe.projectionMatrixInverse.copy(Qe.projectionMatrix).invert(),Qe.viewport.set(mt.x,mt.y,mt.width,mt.height),Ce===0&&(X.matrix.copy(Qe.matrix),X.matrix.decompose(X.position,X.quaternion,X.scale)),Te===!0&&X.cameras.push(Qe)}let pe=s.enabledFeatures;if(pe&&pe.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&A){p=i.getBinding();let Ce=p.getDepthInformation(xe[0]);Ce&&Ce.isValid&&Ce.texture&&g.init(Ce,s.renderState)}if(pe&&pe.includes("camera-access")&&A){e.state.unbindTexture(),p=i.getBinding();for(let Ce=0;Ce<xe.length;Ce++){let st=xe[Ce].camera;if(st){let mt=f[st];mt||(mt=new _c,f[st]=mt);let Qe=p.getCameraImage(st);mt.sourceTexture=Qe}}}}for(let xe=0;xe<T.length;xe++){let Te=b[xe],pe=T[xe];Te!==null&&pe!==void 0&&pe.update(Te,de,c||a)}ee&&ee(Q,de),de.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:de}),x=null}let Ie=new c1;Ie.setAnimationLoop(_e),this.setAnimationLoop=function(Q){ee=Q},this.dispose=function(){}}},I3=new Ft,y1=new Ne;y1.set(-1,0,0,0,1,0,0,0,1);function B3(t,e){function n(g,f){g.matrixAutoUpdate===!0&&g.updateMatrix(),f.value.copy(g.matrix)}function i(g,f){f.color.getRGB(g.fogColor.value,Rv(t)),f.isFog?(g.fogNear.value=f.near,g.fogFar.value=f.far):f.isFogExp2&&(g.fogDensity.value=f.density)}function s(g,f,m,y,_){f.isNodeMaterial?f.uniformsNeedUpdate=!1:f.isMeshBasicMaterial?r(g,f):f.isMeshLambertMaterial?(r(g,f),f.envMap&&(g.envMapIntensity.value=f.envMapIntensity)):f.isMeshToonMaterial?(r(g,f),p(g,f)):f.isMeshPhongMaterial?(r(g,f),h(g,f),f.envMap&&(g.envMapIntensity.value=f.envMapIntensity)):f.isMeshStandardMaterial?(r(g,f),u(g,f),f.isMeshPhysicalMaterial&&d(g,f,_)):f.isMeshMatcapMaterial?(r(g,f),x(g,f)):f.isMeshDepthMaterial?r(g,f):f.isMeshDistanceMaterial?(r(g,f),A(g,f)):f.isMeshNormalMaterial?r(g,f):f.isLineBasicMaterial?(a(g,f),f.isLineDashedMaterial&&o(g,f)):f.isPointsMaterial?l(g,f,m,y):f.isSpriteMaterial?c(g,f):f.isShadowMaterial?(g.color.value.copy(f.color),g.opacity.value=f.opacity):f.isShaderMaterial&&(f.uniformsNeedUpdate=!1)}function r(g,f){g.opacity.value=f.opacity,f.color&&g.diffuse.value.copy(f.color),f.emissive&&g.emissive.value.copy(f.emissive).multiplyScalar(f.emissiveIntensity),f.map&&(g.map.value=f.map,n(f.map,g.mapTransform)),f.alphaMap&&(g.alphaMap.value=f.alphaMap,n(f.alphaMap,g.alphaMapTransform)),f.bumpMap&&(g.bumpMap.value=f.bumpMap,n(f.bumpMap,g.bumpMapTransform),g.bumpScale.value=f.bumpScale,f.side===Qt&&(g.bumpScale.value*=-1)),f.normalMap&&(g.normalMap.value=f.normalMap,n(f.normalMap,g.normalMapTransform),g.normalScale.value.copy(f.normalScale),f.side===Qt&&g.normalScale.value.negate()),f.displacementMap&&(g.displacementMap.value=f.displacementMap,n(f.displacementMap,g.displacementMapTransform),g.displacementScale.value=f.displacementScale,g.displacementBias.value=f.displacementBias),f.emissiveMap&&(g.emissiveMap.value=f.emissiveMap,n(f.emissiveMap,g.emissiveMapTransform)),f.specularMap&&(g.specularMap.value=f.specularMap,n(f.specularMap,g.specularMapTransform)),f.alphaTest>0&&(g.alphaTest.value=f.alphaTest);let m=e.get(f),y=m.envMap,_=m.envMapRotation;y&&(g.envMap.value=y,g.envMapRotation.value.setFromMatrix4(I3.makeRotationFromEuler(_)).transpose(),y.isCubeTexture&&y.isRenderTargetTexture===!1&&g.envMapRotation.value.premultiply(y1),g.reflectivity.value=f.reflectivity,g.ior.value=f.ior,g.refractionRatio.value=f.refractionRatio),f.lightMap&&(g.lightMap.value=f.lightMap,g.lightMapIntensity.value=f.lightMapIntensity,n(f.lightMap,g.lightMapTransform)),f.aoMap&&(g.aoMap.value=f.aoMap,g.aoMapIntensity.value=f.aoMapIntensity,n(f.aoMap,g.aoMapTransform))}function a(g,f){g.diffuse.value.copy(f.color),g.opacity.value=f.opacity,f.map&&(g.map.value=f.map,n(f.map,g.mapTransform))}function o(g,f){g.dashSize.value=f.dashSize,g.totalSize.value=f.dashSize+f.gapSize,g.scale.value=f.scale}function l(g,f,m,y){g.diffuse.value.copy(f.color),g.opacity.value=f.opacity,g.size.value=f.size*m,g.scale.value=y*.5,f.map&&(g.map.value=f.map,n(f.map,g.uvTransform)),f.alphaMap&&(g.alphaMap.value=f.alphaMap,n(f.alphaMap,g.alphaMapTransform)),f.alphaTest>0&&(g.alphaTest.value=f.alphaTest)}function c(g,f){g.diffuse.value.copy(f.color),g.opacity.value=f.opacity,g.rotation.value=f.rotation,f.map&&(g.map.value=f.map,n(f.map,g.mapTransform)),f.alphaMap&&(g.alphaMap.value=f.alphaMap,n(f.alphaMap,g.alphaMapTransform)),f.alphaTest>0&&(g.alphaTest.value=f.alphaTest)}function h(g,f){g.specular.value.copy(f.specular),g.shininess.value=Math.max(f.shininess,1e-4)}function p(g,f){f.gradientMap&&(g.gradientMap.value=f.gradientMap)}function u(g,f){g.metalness.value=f.metalness,f.metalnessMap&&(g.metalnessMap.value=f.metalnessMap,n(f.metalnessMap,g.metalnessMapTransform)),g.roughness.value=f.roughness,f.roughnessMap&&(g.roughnessMap.value=f.roughnessMap,n(f.roughnessMap,g.roughnessMapTransform)),f.envMap&&(g.envMapIntensity.value=f.envMapIntensity)}function d(g,f,m){g.ior.value=f.ior,f.diffuseRoughness>0&&(g.diffuseRoughness.value=f.diffuseRoughness,f.diffuseRoughnessMap&&(g.diffuseRoughnessMap.value=f.diffuseRoughnessMap,n(f.diffuseRoughnessMap,g.diffuseRoughnessMapTransform))),f.sheen>0&&(g.sheenColor.value.copy(f.sheenColor).multiplyScalar(f.sheen),g.sheenRoughness.value=f.sheenRoughness,f.sheenColorMap&&(g.sheenColorMap.value=f.sheenColorMap,n(f.sheenColorMap,g.sheenColorMapTransform)),f.sheenRoughnessMap&&(g.sheenRoughnessMap.value=f.sheenRoughnessMap,n(f.sheenRoughnessMap,g.sheenRoughnessMapTransform))),f.clearcoat>0&&(g.clearcoat.value=f.clearcoat,g.clearcoatRoughness.value=f.clearcoatRoughness,f.clearcoatMap&&(g.clearcoatMap.value=f.clearcoatMap,n(f.clearcoatMap,g.clearcoatMapTransform)),f.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=f.clearcoatRoughnessMap,n(f.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),f.clearcoatNormalMap&&(g.clearcoatNormalMap.value=f.clearcoatNormalMap,n(f.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(f.clearcoatNormalScale),f.side===Qt&&g.clearcoatNormalScale.value.negate())),f.dispersion>0&&(g.dispersion.value=f.dispersion),f.retroreflectivity>0&&(g.retroreflectivity.value=f.retroreflectivity),f.iridescence>0&&(g.iridescence.value=f.iridescence,g.iridescenceIOR.value=f.iridescenceIOR,g.iridescenceThicknessMinimum.value=f.iridescenceThicknessMinimum,g.iridescenceThicknessMaximum.value=f.iridescenceThicknessMaximum,f.iridescenceMap&&(g.iridescenceMap.value=f.iridescenceMap,n(f.iridescenceMap,g.iridescenceMapTransform)),f.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=f.iridescenceThicknessMap,n(f.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),f.transmission>0&&(g.transmission.value=f.transmission,g.transmissionSamplerMap.value=m.texture,g.transmissionSamplerSize.value.set(m.width,m.height),f.transmissionMap&&(g.transmissionMap.value=f.transmissionMap,n(f.transmissionMap,g.transmissionMapTransform)),g.thickness.value=f.thickness,f.thicknessMap&&(g.thicknessMap.value=f.thicknessMap,n(f.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=f.attenuationDistance,g.attenuationColor.value.copy(f.attenuationColor)),f.anisotropy>0&&(g.anisotropyVector.value.set(f.anisotropy*Math.cos(f.anisotropyRotation),f.anisotropy*Math.sin(f.anisotropyRotation)),f.anisotropyMap&&(g.anisotropyMap.value=f.anisotropyMap,n(f.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=f.specularIntensity,g.specularColor.value.copy(f.specularColor),f.specularColorMap&&(g.specularColorMap.value=f.specularColorMap,n(f.specularColorMap,g.specularColorMapTransform)),f.specularIntensityMap&&(g.specularIntensityMap.value=f.specularIntensityMap,n(f.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,f){f.matcap&&(g.matcap.value=f.matcap)}function A(g,f){let m=e.get(f).light;g.referencePosition.value.setFromMatrixPosition(m.matrixWorld),g.nearDistance.value=m.shadow.camera.near,g.farDistance.value=m.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function N3(t,e,n,i){let s={},r={};function a(m,y){let _=y.program;i.uniformBlockBinding(m,_,s[m.id])}function o(m,y){let _=s[m.id];_===void 0&&(x(m),_=l(m),s[m.id]=_,m.addEventListener("dispose",g));let T=y.program;i.updateUBOMapping(m,T);let b=e.render.frame;r[m.id]!==b&&(c(m),r[m.id]=b)}function l(m){let y=t.createBuffer(),_=m.__size,T=m.usage;return t.bindBuffer(t.UNIFORM_BUFFER,y),t.bufferData(t.UNIFORM_BUFFER,_,T),t.bindBuffer(t.UNIFORM_BUFFER,null),y}function c(m){let y=s[m.id],_=m.uniforms,T=m.__cache;t.bindBuffer(t.UNIFORM_BUFFER,y);for(let b=0,v=_.length;b<v;b++){let E=_[b];if(Array.isArray(E))for(let N=0,C=E.length;N<C;N++)h(E[N],b,N,T);else h(E,b,0,T)}t.bindBuffer(t.UNIFORM_BUFFER,null)}function h(m,y,_,T){if(u(m,y,_,T)===!0){let b=m.__offset,v=m.value;if(Array.isArray(v)){let E=0;for(let N=0;N<v.length;N++){let C=v[N],B=A(C,!0);p(C,m.__data,E),E+=B.storage/Float32Array.BYTES_PER_ELEMENT}}else p(v,m.__data,0);t.bufferSubData(t.UNIFORM_BUFFER,b,m.__data)}}function p(m,y,_){typeof m=="number"||typeof m=="boolean"?y[_]=m:m.isMatrix3?(y[_]=m.elements[0],y[_+1]=m.elements[1],y[_+2]=m.elements[2],y[_+3]=0,y[_+4]=m.elements[3],y[_+5]=m.elements[4],y[_+6]=m.elements[5],y[_+7]=0,y[_+8]=m.elements[6],y[_+9]=m.elements[7],y[_+10]=m.elements[8],y[_+11]=0):ArrayBuffer.isView(m)?y.set(m.subarray(0,y.length-_),_):m.toArray(y,_)}function u(m,y,_,T){let b=m.value,v=y+"_"+_;if(Array.isArray(b)){T[v]===void 0&&(T[v]=[]);let E=T[v],N=!1;for(let C=0;C<b.length;C++)d(b[C],E,C)===!0&&(N=!0);return N}return d(b,T,v)}function d(m,y,_){if(y[_]===void 0)return typeof m=="number"||typeof m=="boolean"?y[_]=m:ArrayBuffer.isView(m)?y[_]=m.slice():y[_]=m.clone(),!0;{let T=y[_];if(typeof m=="number"||typeof m=="boolean"){if(T!==m)return y[_]=m,!0}else{if(ArrayBuffer.isView(m))return!0;if(T.equals(m)===!1)return T.copy(m),!0}}return!1}function x(m){let y=m.uniforms,_=0,T=16;for(let v=0,E=y.length;v<E;v++){let N=Array.isArray(y[v])?y[v]:[y[v]];for(let C=0,B=N.length;C<B;C++){let F=N[C],X=Array.isArray(F.value)?F.value:[F.value],D=Array.isArray(y[v])||Array.isArray(F.value),V=_;for(let H=0,te=X.length;H<te;H++){let k=X[H],$=A(k,D),ee=_%T,_e=ee%$.boundary,Ie=ee+_e;_+=_e,Ie!==0&&T-Ie<$.storage&&(_+=T-Ie),H===0&&(V=_),_+=$.storage}let q=F.type==="int"?Int32Array:F.type==="uint"?Uint32Array:Float32Array;F.__data=new q((_-V)/Float32Array.BYTES_PER_ELEMENT),F.__offset=V}}let b=_%T;return b>0&&(_+=T-b),m.__size=_,m.__cache={},this}function A(m,y){let _={boundary:0,storage:0};return typeof m=="number"||typeof m=="boolean"?(_.boundary=4,_.storage=4):m.isVector2?(_.boundary=8,_.storage=8):m.isVector3||m.isColor?(_.boundary=16,_.storage=12):m.isVector4?(_.boundary=16,_.storage=16):m.isMatrix3?(_.boundary=16,_.storage=48):m.isMatrix4?(_.boundary=16,_.storage=64):m.isTexture?De("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(m)?(_.boundary=16,_.storage=m.byteLength):De("WebGLRenderer: Unsupported uniform value type.",m),y===!0&&(_.boundary=16,_.storage=Math.ceil(_.storage/16)*16),_}function g(m){let y=m.target;y.removeEventListener("dispose",g),t.deleteBuffer(s[y.id]),delete s[y.id],delete r[y.id]}function f(){for(let m in s)t.deleteBuffer(s[m]);s={},r={}}return{bind:a,update:o,dispose:f}}var L3=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),xs=null;function P3(){return xs===null&&(xs=new fh(L3,16,16,zr,Xi),xs.name="DFG_LUT",xs.minFilter=Dt,xs.magFilter=Dt,xs.wrapS=hs,xs.wrapT=hs,xs.generateMipmaps=!1,xs.needsUpdate=!0),xs}var gd=class{constructor(e={}){let{canvas:n=BM(),context:i=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:p=!1,reversedDepthBuffer:u=!1,outputBufferType:d=Gt}=e;this.isWebGLRenderer=!0;let x;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");x=i.getContextAttributes().alpha}else x=a;let A=d,g=new Set([Ph,Lh,Nh]),f=new Set([Gt,Wi,jo,Ys,Ih,Bh]),m=new Uint32Array(4),y=new Int32Array(4),_={textures:[],min:[],max:[],resolution:[]},T=null,b=null,v=[],E=[],N=null;this.domElement=n,this.debug={checkShaderErrors:!0,diagnostics:{keywords:!1},onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=gs,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let C=this,B=!1,F=null,X=null,D=null,V=null;this._outputColorSpace=Tt;let q=0,H=0,te=null,k=-1,$=null,ee=new Lt,_e=new Lt,Ie=null,Q=new Ye(0),de=0,xe=n.width,Te=n.height,pe=1,be=null,qe=null,Ce=new Lt(0,0,xe,Te),st=new Lt(0,0,xe,Te),mt=!1,Qe=new Sa,dn=new hh,Zt=!1,zn=!1,It=new Ft,Qi=new G,nn=new Lt,di={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},pi=!1;function mi(){return te===null?pe:1}let I=i;function gi(M,U){return n.getContext(M,U)}let bt,wt,he,Kt,Re,Le,w,S,P,ne,j,re,ue,Y,Z,oe,R,K,W,le,se,ae,Ee;try{let M={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:h,failIfMajorPerformanceCaveat:p};if("setAttribute"in n&&n.setAttribute("data-engine",`three.js r${"187"}`),n.addEventListener("webglcontextlost",we,!1),n.addEventListener("webglcontextrestored",He,!1),n.addEventListener("webglcontextcreationerror",me,!1),I===null){let U="webgl2";if(I=gi(U,M),I===null)throw gi(U)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}Ge()}catch(M){throw n.removeEventListener("webglcontextlost",we,!1),n.removeEventListener("webglcontextrestored",He,!1),n.removeEventListener("webglcontextcreationerror",me,!1),Be("WebGLRenderer: "+M.message),M}function Ge(){bt=new qR(I),bt.init(),se=new R3(I,bt),wt=new PR(I,bt,e,se),he=new w3(I,bt),wt.reversedDepthBuffer&&u&&he.buffers.depth.setReversed(!0),X=I.createFramebuffer(),D=I.createFramebuffer(),V=I.createFramebuffer(),Kt=new KR(I),Re=new d3,Le=new C3(I,bt,he,Re,wt,se,Kt),w=new YR(C),S=new jw(I),ae=new NR(I,S),P=new QR(I,S,Kt,ae),ne=new jR(I,P,S,ae,Kt),K=new JR(I,wt,Le),Z=new OR(Re),j=new h3(C,w,bt,wt,ae,Z),re=new B3(C,Re),ue=new m3,Y=new S3(bt),R=new BR(C,w,he,ne,x,l),oe=new b3(C,ne,wt),Ee=new N3(I,Kt,wt,he),W=new LR(I,bt,Kt),le=new ZR(I,bt,Kt),Kt.programs=j.programs,C.capabilities=wt,C.extensions=bt,C.properties=Re,C.renderLists=ue,C.shadowMap=oe,C.state=he,C.info=Kt}A!==Gt&&(N=new eD(A,n.width,n.height,o,s,r));let ie=new qv(C,I);this.xr=ie,this.getContext=function(){return I},this.getContextAttributes=function(){return I.getContextAttributes()},this.forceContextLoss=function(){let M=bt.get("WEBGL_lose_context");M&&M.loseContext()},this.forceContextRestore=function(){let M=bt.get("WEBGL_lose_context");M&&M.restoreContext()},this.getPixelRatio=function(){return pe},this.setPixelRatio=function(M){M!==void 0&&(pe=M,this.setSize(xe,Te,!1))},this.getSize=function(M){return M.set(xe,Te)},this.setSize=function(M,U,z=!0){if(ie.isPresenting){De("WebGLRenderer: Can't change size while VR device is presenting.");return}xe=M,Te=U,n.width=Math.floor(M*pe),n.height=Math.floor(U*pe),z===!0&&(n.style.width=M+"px",n.style.height=U+"px"),N!==null&&N.setSize(n.width,n.height),this.setViewport(0,0,M,U)},this.getDrawingBufferSize=function(M){return M.set(xe*pe,Te*pe).floor()},this.setDrawingBufferSize=function(M,U,z){xe=M,Te=U,pe=z,n.width=Math.floor(M*z),n.height=Math.floor(U*z),this.setViewport(0,0,M,U)},this.setEffects=function(M){if(A===Gt){Be("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(M){for(let U=0;U<M.length;U++)if(M[U].isOutputPass===!0){De("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}N.setEffects(M||[])},this.getCurrentViewport=function(M){return M.copy(ee)},this.getViewport=function(M){return M.copy(Ce)},this.setViewport=function(M,U,z,L){M.isVector4?Ce.set(M.x,M.y,M.z,M.w):Ce.set(M,U,z,L),he.viewport(ee.copy(Ce).multiplyScalar(mi()).round())},this.getScissor=function(M){return M.copy(st)},this.setScissor=function(M,U,z,L){M.isVector4?st.set(M.x,M.y,M.z,M.w):st.set(M,U,z,L),he.scissor(_e.copy(st).multiplyScalar(mi()).round())},this.getScissorTest=function(){return mt},this.setScissorTest=function(M){he.setScissorTest(mt=M)},this.setOpaqueSort=function(M){be=M},this.setTransparentSort=function(M){qe=M},this.getClearColor=function(M){return M.copy(R.getClearColor())},this.setClearColor=function(){R.setClearColor(...arguments)},this.getClearAlpha=function(){return R.getClearAlpha()},this.setClearAlpha=function(){R.setClearAlpha(...arguments)},this.clear=function(M=!0,U=!0,z=!0){let L=0;if(M){let O=!1;if(te!==null){let fe=te.texture.format;O=g.has(fe)}if(O){let fe=te.texture.type,ye=f.has(fe),ce=R.getClearColor(),ve=R.getClearAlpha(),Me=ce.r,Fe=ce.g,Ze=ce.b;ye?(m[0]=Me,m[1]=Fe,m[2]=Ze,m[3]=ve,I.clearBufferuiv(I.COLOR,0,m)):(y[0]=Me,y[1]=Fe,y[2]=Ze,y[3]=ve,I.clearBufferiv(I.COLOR,0,y))}else L|=I.COLOR_BUFFER_BIT}U&&(L|=I.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),z&&(L|=I.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),L!==0&&I.clear(L)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(M){M.setRenderer(this),F=M},this.dispose=function(){n.removeEventListener("webglcontextlost",we,!1),n.removeEventListener("webglcontextrestored",He,!1),n.removeEventListener("webglcontextcreationerror",me,!1),R.dispose(),ue.dispose(),Y.dispose(),Le.dispose(),Re.dispose(),w.dispose(),ne.dispose(),P.dispose(),ae.dispose(),Ee.dispose(),j.dispose(),ie.dispose(),ie.removeEventListener("sessionstart",n0),ie.removeEventListener("sessionend",i0),Gr.stop()};function we(M){M.preventDefault(),wv("WebGLRenderer: Context Lost."),B=!0}function He(){wv("WebGLRenderer: Context Restored."),B=!1;let M=Kt.autoReset,U=oe.enabled,z=oe.autoUpdate,L=oe.needsUpdate,O=oe.type;Ge(),Kt.autoReset=M,oe.enabled=U,oe.autoUpdate=z,oe.needsUpdate=L,oe.type=O}function me(M){Be("WebGLRenderer: A WebGL context could not be created. Reason: ",M.statusMessage)}function ut(M){let U=M.target;U.removeEventListener("dispose",ut),sn(U)}function sn(M){qs(M),Re.remove(M)}function qs(M){let U=Re.get(M).programs;U!==void 0&&(U.forEach(function(z){j.releaseProgram(z)}),M.isShaderMaterial&&j.releaseShaderCache(M))}this.renderBufferDirect=function(M,U,z,L,O,fe){U===null&&(U=di);let ye=O.isMesh&&O.matrixWorld.determinantAffine()<0,ce=P1(M,U,z,L,O);he.setMaterial(L,ye);let ve=z.index,Me=1;if(L.wireframe===!0){if(ve=P.getWireframeAttribute(z),ve===void 0)return;Me=2}let Fe=z.drawRange,Ze=z.attributes.position,Se=Fe.start*Me,Ke=(Fe.start+Fe.count)*Me;fe!==null&&(Se=Math.max(Se,fe.start*Me),Ke=Math.min(Ke,(fe.start+fe.count)*Me)),ve!==null?(Se=Math.max(Se,0),Ke=Math.min(Ke,ve.count)):Ze!=null&&(Se=Math.max(Se,0),Ke=Math.min(Ke,Ze.count));let Vt=Ke-Se;if(Vt<0||Vt===1/0)return;ae.setup(O,L,ce,z,ve);let yt,gt=W;if(ve!==null&&(yt=S.get(ve),gt=le,gt.setIndex(yt)),O.isMesh)L.wireframe===!0?(he.setLineWidth(L.wireframeLinewidth*mi()),gt.setMode(I.LINES)):gt.setMode(I.TRIANGLES);else if(O.isLine){let An=L.linewidth;An===void 0&&(An=1),he.setLineWidth(An*mi()),O.isLineSegments?gt.setMode(I.LINES):O.isLineLoop?gt.setMode(I.LINE_LOOP):gt.setMode(I.LINE_STRIP)}else O.isPoints?gt.setMode(I.POINTS):O.isSprite&&gt.setMode(I.TRIANGLES);if(O.isBatchedMesh)if(bt.get("WEBGL_multi_draw"))gt.renderMultiDraw(O._multiDrawStarts,O._multiDrawCounts,O._multiDrawCount);else{let An=O._multiDrawStarts,Ae=O._multiDrawCounts,Un=O._multiDrawCount,rt=ve?S.get(ve).bytesPerElement:1,Vr=Re.get(L).currentProgram.getUniforms();for(let Mn=0;Mn<Un;Mn++)Vr.setValue(I,"_gl_DrawID",Mn),gt.render(An[Mn]/rt,Ae[Mn])}else if(O.isInstancedMesh)gt.renderInstances(Se,Vt,O.count);else if(z.isInstancedBufferGeometry){let An=z._maxInstanceCount!==void 0?z._maxInstanceCount:1/0,Ae=Math.min(z.instanceCount,An);gt.renderInstances(Se,Vt,Ae)}else gt.render(Se,Vt)};function t0(M,U,z,L){F!==null&&M.isNodeMaterial&&F.setObject(L,M),Zt===!0&&Z.setState(M,z,!1),M.transparent===!0&&M.side===Fn&&M.forceSinglePass===!1?(M.side=Qt,M.needsUpdate=!0,tl(M,U,L),M.side=ms,M.needsUpdate=!0,tl(M,U,L),M.side=Fn):tl(M,U,L)}this.compile=function(M,U,z=null){z===null&&(z=M),F!==null&&F.renderStart(M,U,z),b=Y.get(z),b.init(U),E.push(b),z.traverseVisible(function(O){O.isLight&&O.layers.test(U.layers)&&(b.pushLight(O),O.castShadow&&b.pushShadow(O))}),M!==z&&M.traverseVisible(function(O){O.isLight&&O.layers.test(U.layers)&&(b.pushLight(O),O.castShadow&&b.pushShadow(O))}),b.setupLights(),F!==null&&F.updateLights(b.state.lightsArray),zn=this.localClippingEnabled,Zt=Z.init(this.clippingPlanes,zn),Zt===!0&&Z.setGlobalState(this.clippingPlanes,U),F!==null&&oe.render(b.state.shadowsArray,z,U);let L=new Set;return M.traverse(function(O){if(!(O.isMesh||O.isPoints||O.isLine||O.isSprite))return;let fe=O.material;if(fe)if(Array.isArray(fe))for(let ye=0;ye<fe.length;ye++){let ce=fe[ye];t0(ce,z,U,O),L.add(ce)}else t0(fe,z,U,O),L.add(fe)}),b=E.pop(),F!==null&&F.renderEnd(),L},this.compileAsync=function(M,U,z=null){let L=this.compile(M,U,z);return new Promise(O=>{function fe(){if(L.forEach(function(ye){let ve=Re.get(ye).currentProgram;(ve===void 0||ve.isReady())&&L.delete(ye)}),L.size===0){O(M);return}setTimeout(fe,10)}bt.get("KHR_parallel_shader_compile")!==null?fe():setTimeout(fe,10)})};let Md=null;function N1(M){Md&&Md(M)}function n0(){Gr.stop()}function i0(){Gr.start()}let Gr=new c1;Gr.setAnimationLoop(N1),typeof self<"u"&&Gr.setContext(self),this.setAnimationLoop=function(M){Md=M,ie.setAnimationLoop(M),M===null?Gr.stop():Gr.start()},ie.addEventListener("sessionstart",n0),ie.addEventListener("sessionend",i0),this.render=function(M,U){if(U!==void 0&&U.isCamera!==!0){Be("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(B===!0)return;F!==null&&F.renderStart(M,U);let z=ie.enabled===!0&&ie.isPresenting===!0,L=N!==null&&(te===null||z)&&N.begin(C,te);if(M.matrixWorldAutoUpdate===!0&&M.updateMatrixWorld(),U.parent===null&&U.matrixWorldAutoUpdate===!0&&U.updateMatrixWorld(),ie.enabled===!0&&ie.isPresenting===!0&&(N===null||N.isCompositing()===!1)&&(ie.cameraAutoUpdate===!0&&ie.updateCamera(U),U=ie.getCamera()),M.isScene===!0&&M.onBeforeRender(C,M,U,te),b=Y.get(M,E.length),b.init(U),b.state.textureUnits=Le.getTextureUnits(),E.push(b),It.multiplyMatrices(U.projectionMatrix,U.matrixWorldInverse),U.isArrayCamera?dn.setFromArrayCamera(U):Qe.setFromProjectionMatrix(It,Ci,U.reversedDepth),zn=this.localClippingEnabled,Zt=Z.init(this.clippingPlanes,zn),T=ue.get(M,v.length),T.init(),v.push(T),this.info.render.frame++,ie.enabled===!0&&ie.isPresenting===!0){let ye=C.xr.getDepthSensingMesh();ye!==null&&Ed(ye,U,-1/0,C.sortObjects)}Ed(M,U,0,C.sortObjects),T.finish(),F!==null&&F.updateLights(b.state.lightsArray),C.sortObjects===!0&&T.sort(be,qe),pi=ie.enabled===!1||ie.isPresenting===!1||ie.hasDepthSensing()===!1,pi&&R.addToRenderList(T,M),this.info.autoReset===!0&&this.info.reset(),Zt===!0&&Z.beginShadows();let O=b.state.shadowsArray;if(oe.render(O,M,U),Zt===!0&&Z.endShadows(),(L&&N.hasRenderPass())===!1){let ye=T.opaque,ce=T.transmissive;if(b.setupLights(),U.isArrayCamera){let ve=U.cameras;if(ce.length>0)for(let Me=0,Fe=ve.length;Me<Fe;Me++){let Ze=ve[Me];r0(ye,ce,M,Ze)}pi&&R.render(M);for(let Me=0,Fe=ve.length;Me<Fe;Me++){let Ze=ve[Me];s0(T,M,Ze,Ze.viewport)}}else ce.length>0&&r0(ye,ce,M,U),pi&&R.render(M),s0(T,M,U)}te!==null&&H===0&&(Le.updateMultisampleRenderTarget(te),Le.updateRenderTargetMipmap(te)),L&&N.end(C),M.isScene===!0&&M.onAfterRender(C,M,U),ae.resetDefaultState(),k=-1,$=null,E.pop(),E.length>0?(b=E[E.length-1],Le.setTextureUnits(b.state.textureUnits),Zt===!0&&Z.setGlobalState(C.clippingPlanes,b.state.camera)):b=null,v.pop(),v.length>0?T=v[v.length-1]:T=null,F!==null&&F.renderEnd()};function Ed(M,U,z,L){if(M.visible===!1)return;if(M.layers.test(U.layers)){if(M.isGroup)z=M.renderOrder;else if(M.isLOD)M.autoUpdate===!0&&M.update(U);else if(M.isLight)b.pushLight(M),M.castShadow&&b.pushShadow(M);else if(M.isSprite){let ye=U.isArrayCamera?dn:Qe;if(!M.frustumCulled||M.intersectsFrustum(ye)){L&&nn.setFromMatrixPosition(M.matrixWorld).applyMatrix4(It);let ce=ne.update(M),ve=M.material;ve.visible&&T.push(M,ce,ve,z,nn.z,null,U)}}else if(M.isMesh||M.isLine||M.isPoints){let ye=U.isArrayCamera?dn:Qe;if(!M.frustumCulled||M.intersectsFrustum(ye)){let ce=ne.update(M),ve=M.material;if(L&&(M.boundingSphere!==void 0?(M.boundingSphere===null&&M.computeBoundingSphere(),nn.copy(M.boundingSphere.center)):(ce.boundingSphere===null&&ce.computeBoundingSphere(),nn.copy(ce.boundingSphere.center)),nn.applyMatrix4(M.matrixWorld).applyMatrix4(It)),Array.isArray(ve)){let Me=ce.groups;for(let Fe=0,Ze=Me.length;Fe<Ze;Fe++){let Se=Me[Fe],Ke=ve[Se.materialIndex];Ke&&Ke.visible&&T.push(M,ce,Ke,z,nn.z,Se,U)}}else ve.visible&&T.push(M,ce,ve,z,nn.z,null,U)}}}let fe=M.children;for(let ye=0,ce=fe.length;ye<ce;ye++)Ed(fe[ye],U,z,L)}function s0(M,U,z,L){let{opaque:O,transmissive:fe,transparent:ye}=M;b.setupLightsView(z),Zt===!0&&Z.setGlobalState(C.clippingPlanes,z),L&&he.viewport(ee.copy(L)),O.length>0&&zc(O,U,z),fe.length>0&&zc(fe,U,z),ye.length>0&&zc(ye,U,z),he.buffers.depth.setTest(!0),he.buffers.depth.setMask(!0),he.buffers.color.setMask(!0),he.setPolygonOffset(!1)}function r0(M,U,z,L){if((z.isScene===!0?z.overrideMaterial:null)!==null)return;if(b.state.transmissionRenderTarget[L.id]===void 0){let Se=bt.has("EXT_color_buffer_half_float")||bt.has("EXT_color_buffer_float");b.state.transmissionRenderTarget[L.id]=new zt(1,1,{generateMipmaps:!0,type:Se?Xi:Gt,minFilter:Xs,samples:Math.max(4,wt.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,colorSpace:je.workingColorSpace})}let fe=b.state.transmissionRenderTarget[L.id],ye=L.viewport||ee;fe.setSize(ye.z*C.transmissionResolutionScale,ye.w*C.transmissionResolutionScale);let ce=C.getRenderTarget(),ve=C.getActiveCubeFace(),Me=C.getActiveMipmapLevel();C.setRenderTarget(fe),C.getClearColor(Q),de=C.getClearAlpha(),de<1&&C.setClearColor(16777215,.5),C.clear(),pi&&R.render(z);let Fe=C.toneMapping;C.toneMapping=gs;let Ze=L.viewport;if(L.viewport!==void 0&&(L.viewport=void 0),b.setupLightsView(L),Zt===!0&&Z.setGlobalState(C.clippingPlanes,L),zc(M,z,L),Le.updateMultisampleRenderTarget(fe),Le.updateRenderTargetMipmap(fe),bt.has("WEBGL_multisampled_render_to_texture")===!1){let Se=!1;for(let Ke=0,Vt=U.length;Ke<Vt;Ke++){let yt=U[Ke],{object:gt,geometry:An,material:Ae,group:Un}=yt;if(Ae.side===Fn&&gt.layers.test(L.layers)){let rt=Ae.side;Ae.side=Qt,Ae.needsUpdate=!0,a0(gt,z,L,An,Ae,Un),Ae.side=rt,Ae.needsUpdate=!0,Se=!0}}Se===!0&&(Le.updateMultisampleRenderTarget(fe),Le.updateRenderTargetMipmap(fe))}C.setRenderTarget(ce,ve,Me),C.setClearColor(Q,de),Ze!==void 0&&(L.viewport=Ze),C.toneMapping=Fe}function zc(M,U,z){let L=U.isScene===!0?U.overrideMaterial:null;for(let O=0,fe=M.length;O<fe;O++){let ye=M[O],{object:ce,geometry:ve,group:Me}=ye,Fe=ye.material;Fe.allowOverride===!0&&L!==null&&(Fe=L),ce.layers.test(z.layers)&&a0(ce,U,z,ve,Fe,Me)}}function a0(M,U,z,L,O,fe){F!==null&&O.isNodeMaterial&&F.setObject(M,O),M.onBeforeRender(C,U,z,L,O,fe),M.modelViewMatrix.multiplyMatrices(z.matrixWorldInverse,M.matrixWorld),M.normalMatrix.getNormalMatrix(M.modelViewMatrix),O.onBeforeRender(C,U,z,L,M,fe),O.transparent===!0&&O.side===Fn&&O.forceSinglePass===!1?(O.side=Qt,O.needsUpdate=!0,C.renderBufferDirect(z,U,L,O,M,fe),O.side=ms,O.needsUpdate=!0,C.renderBufferDirect(z,U,L,O,M,fe),O.side=Fn):C.renderBufferDirect(z,U,L,O,M,fe),M.onAfterRender(C,U,z,L,O,fe)}function o0(M){let U=0;for(let z of M)z.texture!==null&&U++;return U}function tl(M,U,z){U.isScene!==!0&&(U=di);let L=Re.get(M),O=b.state.lights,fe=b.state.shadowsArray,ye=O.state.version,ce=j.getParameters(M,O.state,fe,U,z,b.state.lightProbeGridArray),ve=j.getProgramCacheKey(ce),Me=L.programs;L.environment=M.isMeshStandardMaterial||M.isMeshLambertMaterial||M.isMeshPhongMaterial?U.environment:null,L.fog=U.fog;let Fe=M.isMeshStandardMaterial||M.isMeshLambertMaterial&&!M.envMap||M.isMeshPhongMaterial&&!M.envMap;L.envMap=w.get(M.envMap||L.environment,Fe),L.envMapRotation=L.environment!==null&&M.envMap===null?U.environmentRotation:M.envMapRotation,Me===void 0&&(M.addEventListener("dispose",ut),Me=new Map,L.programs=Me,L.programVariants=new Map);let Ze=Me.get(ve);if(Ze!==void 0){if(L.currentProgram===Ze&&L.lightsStateVersion===ye)return c0(M,ce),Ze}else ce.uniforms=j.getUniforms(M),F!==null&&M.isNodeMaterial&&F.build(M,z,ce),M.onBeforeCompile(ce,C),Ze=j.acquireProgram(ce,ve),Me.set(ve,Ze),L.uniforms=ce.uniforms;let Se=L.uniforms;return(!M.isShaderMaterial&&!M.isRawShaderMaterial||M.clipping===!0)&&(Se.clippingPlanes=Z.uniform),c0(M,ce),L.needsLights=F1(M),L.lightsStateVersion=ye,L.needsLights&&(Se.ambientLightColor.value=O.state.ambient,Se.lightProbe.value=O.state.probe,Se.sunLights.value=O.state.sun,Se.sunLightShadows.value=O.state.sunShadow,Se.directionalLights.value=O.state.directional,Se.directionalLightShadows.value=O.state.directionalShadow,Se.spotLights.value=O.state.spot,Se.spotLightShadows.value=O.state.spotShadow,Se.rectAreaLights.value=O.state.rectArea,Se.ltc_1.value=O.state.rectAreaLTC1,Se.ltc_2.value=O.state.rectAreaLTC2,Se.pointLights.value=O.state.point,Se.pointLightShadows.value=O.state.pointShadow,Se.hemisphereLights.value=O.state.hemi,Se.sunShadowMatrix.value=O.state.sunShadowMatrix,Se.sunShadowCascade.value=O.state.sunShadowCascade,Se.directionalShadowMatrix.value=O.state.directionalShadowMatrix,Se.spotLightMatrix.value=O.state.spotLightMatrix,Se.spotLightMap.value=O.state.spotLightMap,Se.pointShadowMatrix.value=O.state.pointShadowMatrix),L.numLightProbeGrids=o0(b.state.lightProbeGridArray),L.currentProgram=Ze,L.uniformsList=null,Ze}function L1(M,U,z,L,O){let fe=Re.get(M),ye=fe.currentProgram.id+","+L+","+O,ce=fe.programVariants.get(ye);return ce===void 0?(ce=tl(M,U,z),fe.programVariants.set(ye,ce)):(fe.currentProgram=ce,fe.uniformsList=null,fe.outputColorSpace=L,fe.toneMapping=O),ce}function l0(M){if(M.uniformsList===null){let U=M.currentProgram.getUniforms();M.uniformsList=$o.seqWithValue(U.seq,M.uniforms)}return M.uniformsList}function c0(M,U){let z=Re.get(M);z.outputColorSpace=U.outputColorSpace,z.batching=U.batching,z.batchingColor=U.batchingColor,z.instancing=U.instancing,z.instancingColor=U.instancingColor,z.instancingMorph=U.instancingMorph,z.skinning=U.skinning,z.morphTargets=U.morphTargets,z.morphNormals=U.morphNormals,z.morphColors=U.morphColors,z.morphTargetsCount=U.morphTargetsCount,z.numClippingPlanes=U.numClippingPlanes,z.numIntersection=U.numClipIntersection,z.vertexAlphas=U.vertexAlphas,z.vertexTangents=U.vertexTangents,z.toneMapping=U.toneMapping}function P1(M,U,z,L,O){U.isScene!==!0&&(U=di),Le.resetTextureUnits();let fe=U.fog,ye=L.isMeshStandardMaterial||L.isMeshLambertMaterial||L.isMeshPhongMaterial?U.environment:null,ce=te===null?C.outputColorSpace:te.isXRRenderTarget===!0?te.texture.colorSpace:je.workingColorSpace,ve=L.isMeshStandardMaterial||L.isMeshLambertMaterial&&!L.envMap||L.isMeshPhongMaterial&&!L.envMap,Me=w.get(L.envMap||ye,ve),Fe=L.vertexColors===!0&&!!z.attributes.color&&z.attributes.color.itemSize===4,Ze=!!z.attributes.tangent&&(!!L.normalMap||L.anisotropy>0),Se=!!z.morphAttributes.position,Ke=!!z.morphAttributes.normal,Vt=!!z.morphAttributes.color,yt=gs;L.toneMapped&&(te===null||te.isXRRenderTarget===!0)&&(yt=C.toneMapping);let gt=z.morphAttributes.position||z.morphAttributes.normal||z.morphAttributes.color,An=gt!==void 0?gt.length:0,Ae=Re.get(L),Un=b.state.lights;if(Zt===!0&&(zn===!0||M!==$)){let Bt=M===$&&L.id===k;Z.setState(L,M,Bt)}let rt=!1,Vr=!1;L.version===Ae.__version?Ae.needsLights&&Ae.lightsStateVersion!==Un.state.version||O.isBatchedMesh&&Ae.batching===!1||!O.isBatchedMesh&&Ae.batching===!0||O.isBatchedMesh&&Ae.batchingColor===!0&&O._colorsTexture===null||O.isBatchedMesh&&Ae.batchingColor===!1&&O._colorsTexture!==null||O.isInstancedMesh&&Ae.instancing===!1||!O.isInstancedMesh&&Ae.instancing===!0||O.isSkinnedMesh&&Ae.skinning===!1||!O.isSkinnedMesh&&Ae.skinning===!0||O.isInstancedMesh&&Ae.instancingColor===!0&&O.instanceColor===null||O.isInstancedMesh&&Ae.instancingColor===!1&&O.instanceColor!==null||O.isInstancedMesh&&Ae.instancingMorph===!0&&O.morphTexture===null||O.isInstancedMesh&&Ae.instancingMorph===!1&&O.morphTexture!==null||Ae.envMap!==Me||L.fog===!0&&Ae.fog!==fe||Ae.numClippingPlanes!==void 0&&(Ae.numClippingPlanes!==Z.numPlanes||Ae.numIntersection!==Z.numIntersection)||Ae.vertexAlphas!==Fe||Ae.vertexTangents!==Ze||Ae.morphTargets!==Se||Ae.morphNormals!==Ke||Ae.morphColors!==Vt||Ae.morphTargetsCount!==An||Ae.numLightProbeGrids!==o0(b.state.lightProbeGridArray)?rt=!0:(Ae.outputColorSpace!==ce||Ae.toneMapping!==yt)&&(Vr=!0):(rt=!0,Ae.__version=L.version);let Mn=Ae.currentProgram;(rt===!0||Vr===!0)&&(Mn=Vr===!0?L1(L,U,O,ce,yt):tl(L,U,O),F&&L.isNodeMaterial&&F.onUpdateProgram(L,Mn,Ae));let Hc=!1,Qs=!1,kr=!1,tt=Mn.getUniforms(),pn=Ae.uniforms;if(he.useProgram(Mn.program)&&(Hc=!0,Qs=!0,kr=!0),L.id!==k&&(k=L.id,Qs=!0),Hc||$!==M){he.buffers.depth.getReversed()&&M.reversedDepth!==!0&&(M._reversedDepth=!0,M.updateProjectionMatrix()),tt.setValue(I,"projectionMatrix",M.projectionMatrix),tt.setValue(I,"viewMatrix",M.matrixWorldInverse);let rn=tt.map.cameraPosition;rn!==void 0&&rn.setValue(I,Qi.setFromMatrixPosition(M.matrixWorld)),wt.logarithmicDepthBuffer&&tt.setValue(I,"logDepthBufFC",2/(Math.log(M.far+1)/Math.LN2)),(L.isMeshPhongMaterial||L.isMeshToonMaterial||L.isMeshLambertMaterial||L.isMeshBasicMaterial||L.isMeshStandardMaterial||L.isShaderMaterial)&&tt.setValue(I,"isOrthographic",M.isOrthographicCamera===!0),$!==M&&($=M,Qs=!0,kr=!0)}if(Ae.needsLights&&(Un.state.sunShadowMap.length>0&&tt.setValue(I,"sunShadowMap",Un.state.sunShadowMap,Le),Un.state.directionalShadowMap.length>0&&tt.setValue(I,"directionalShadowMap",Un.state.directionalShadowMap,Le),Un.state.spotShadowMap.length>0&&tt.setValue(I,"spotShadowMap",Un.state.spotShadowMap,Le),Un.state.pointShadowMap.length>0&&tt.setValue(I,"pointShadowMap",Un.state.pointShadowMap,Le)),O.isSkinnedMesh){tt.setOptional(I,O,"bindMatrix"),tt.setOptional(I,O,"bindMatrixInverse");let Bt=O.skeleton;Bt&&(Bt.boneTexture===null&&Bt.computeBoneTexture(),tt.setValue(I,"boneTexture",Bt.boneTexture,Le))}O.isBatchedMesh&&(tt.setOptional(I,O,"batchingTexture"),tt.setValue(I,"batchingTexture",O._matricesTexture,Le),tt.setOptional(I,O,"batchingIdTexture"),tt.setValue(I,"batchingIdTexture",O._indirectTexture,Le),tt.setOptional(I,O,"batchingColorTexture"),O._colorsTexture!==null&&tt.setValue(I,"batchingColorTexture",O._colorsTexture,Le));let Ui=z.morphAttributes;if((Ui.position!==void 0||Ui.normal!==void 0||Ui.color!==void 0)&&K.update(O,z,Mn),(Qs||Ae.receiveShadow!==O.receiveShadow)&&(Ae.receiveShadow=O.receiveShadow,tt.setValue(I,"receiveShadow",O.receiveShadow)),(L.isMeshStandardMaterial||L.isMeshLambertMaterial||L.isMeshPhongMaterial)&&L.envMap===null&&U.environment!==null&&(pn.envMapIntensity.value=U.environmentIntensity),pn.dfgLUT!==void 0&&(pn.dfgLUT.value=P3()),Qs){if(tt.setValue(I,"toneMappingExposure",C.toneMappingExposure),Ae.needsLights&&O1(pn,kr),fe&&L.fog===!0&&re.refreshFogUniforms(pn,fe),re.refreshMaterialUniforms(pn,L,pe,Te,b.state.transmissionRenderTarget[M.id]),Ae.needsLights&&Ae.numLightProbeGrids>0){let Bt=_,rn=0;for(let _s of b.state.lightProbeGridArray)_s.texture!==null&&(Bt.textures[rn]=_s.texture,Bt.min[rn]=_s.boundingBox.min,Bt.max[rn]=_s.boundingBox.max,Bt.resolution[rn]=_s.resolution,rn++);for(let _s in Bt)Bt[_s].length=rn;pn.probesSH.value=Bt.textures,pn.probesMin.value=Bt.min,pn.probesMax.value=Bt.max,pn.probesResolution.value=Bt.resolution}$o.upload(I,l0(Ae),pn,Le)}if(L.isShaderMaterial&&L.uniformsNeedUpdate===!0&&($o.upload(I,l0(Ae),pn,Le),L.uniformsNeedUpdate=!1),L.isSpriteMaterial&&tt.setValue(I,"center",O.center),tt.setValue(I,"modelViewMatrix",O.modelViewMatrix),tt.setValue(I,"normalMatrix",O.normalMatrix),tt.setValue(I,"modelMatrix",O.matrixWorld),L.uniformsGroups!==void 0){let Bt=L.uniformsGroups;for(let rn=0,_s=Bt.length;rn<_s;rn++){let f0=Bt[rn];Ee.update(f0,Mn),Ee.bind(f0,Mn)}}return Mn}function O1(M,U){M.ambientLightColor.needsUpdate=U,M.lightProbe.needsUpdate=U,M.sunLights.needsUpdate=U,M.sunLightShadows.needsUpdate=U,M.directionalLights.needsUpdate=U,M.directionalLightShadows.needsUpdate=U,M.pointLights.needsUpdate=U,M.pointLightShadows.needsUpdate=U,M.spotLights.needsUpdate=U,M.spotLightShadows.needsUpdate=U,M.rectAreaLights.needsUpdate=U,M.hemisphereLights.needsUpdate=U}function F1(M){return M.isMeshLambertMaterial||M.isMeshToonMaterial||M.isMeshPhongMaterial||M.isMeshStandardMaterial||M.isShadowMaterial||M.isShaderMaterial&&M.lights===!0}this.getActiveCubeFace=function(){return q},this.getActiveMipmapLevel=function(){return H},this.getRenderTarget=function(){return te},this.setRenderTargetTextures=function(M,U,z){let L=Re.get(M);L.__autoAllocateDepthBuffer=M.resolveDepthBuffer===!1,L.__autoAllocateDepthBuffer===!1&&(L.__useRenderToTexture=!1),Re.get(M.texture).__webglTexture=U,Re.get(M.depthTexture).__webglTexture=L.__autoAllocateDepthBuffer?void 0:z,L.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(M,U){let z=Re.get(M);z.__webglFramebuffer=U,z.__useDefaultFramebuffer=U===void 0},this.setRenderTarget=function(M,U=0,z=0){te=M,q=U,H=z;let L=null,O=!1,fe=!1;if(M){let ce=Re.get(M);if(ce.__useDefaultFramebuffer!==void 0){he.bindFramebuffer(I.FRAMEBUFFER,ce.__webglFramebuffer),ee.copy(M.viewport),_e.copy(M.scissor),Ie=M.scissorTest,he.viewport(ee),he.scissor(_e),he.setScissorTest(Ie),k=-1;return}else if(ce.__webglFramebuffer===void 0)Le.setupRenderTarget(M);else if(ce.__hasExternalTextures)Le.rebindTextures(M,Re.get(M.texture).__webglTexture,Re.get(M.depthTexture).__webglTexture);else if(M.depthBuffer){let Fe=M.depthTexture;if(ce.__boundDepthTexture!==Fe){if(Fe!==null&&Re.has(Fe)&&(M.width!==Fe.image.width||M.height!==Fe.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");Le.setupDepthRenderbuffer(M)}}let ve=M.texture;(ve.isData3DTexture||ve.isDataArrayTexture||ve.isCompressedArrayTexture)&&(fe=!0);let Me=Re.get(M).__webglFramebuffer;M.isWebGLCubeRenderTarget?(Array.isArray(Me[U])?L=Me[U][z]:L=Me[U],O=!0):M.samples>0&&Le.useMultisampledRTT(M)===!1?L=Re.get(M).__webglMultisampledFramebuffer:Array.isArray(Me)?L=Me[z]:L=Me,ee.copy(M.viewport),_e.copy(M.scissor),Ie=M.scissorTest}else ee.copy(Ce).multiplyScalar(pe).floor(),_e.copy(st).multiplyScalar(pe).floor(),Ie=mt;if(z!==0&&(L=X),he.bindFramebuffer(I.FRAMEBUFFER,L)&&he.drawBuffers(M,L),he.viewport(ee),he.scissor(_e),he.setScissorTest(Ie),O){let ce=Re.get(M.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_CUBE_MAP_POSITIVE_X+U,ce.__webglTexture,z)}else if(fe){let ce=U;for(let ve=0;ve<M.textures.length;ve++){let Me=Re.get(M.textures[ve]);I.framebufferTextureLayer(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0+ve,Me.__webglTexture,z,ce)}}else if(M!==null&&z!==0){let ce=Re.get(M.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,ce.__webglTexture,z)}k=-1};function u0(M){let U=Re.get(M);return(U.__readFormat!==M.format||U.__readType!==M.type)&&(U.__readFormat=M.format,U.__readType=M.type,U.__formatReadable=wt.textureFormatReadable(M.format),U.__typeReadable=wt.textureTypeReadable(M.type)),U}this.readRenderTargetPixels=function(M,U,z,L,O,fe,ye,ce=0){if(!(M&&M.isWebGLRenderTarget)){Be("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ve=Re.get(M).__webglFramebuffer;if(M.isWebGLCubeRenderTarget&&ye!==void 0&&(ve=ve[ye],Array.isArray(ve)&&(ve=ve[0])),ve){he.bindFramebuffer(I.FRAMEBUFFER,ve);try{let Me=M.textures[ce],Fe=Me.format,Ze=Me.type;M.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+ce);let Se=u0(Me);if(Se.__formatReadable===!1){Be("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(Se.__typeReadable===!1){Be("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}U>=0&&U<=M.width-L&&z>=0&&z<=M.height-O&&I.readPixels(U,z,L,O,se.convert(Fe),se.convert(Ze),fe)}finally{let Me=te!==null?Re.get(te).__webglFramebuffer:null;he.bindFramebuffer(I.FRAMEBUFFER,Me)}}},this.readRenderTargetPixelsAsync=async function(M,U,z,L,O,fe,ye,ce=0){if(!(M&&M.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ve=Re.get(M).__webglFramebuffer;if(M.isWebGLCubeRenderTarget&&ye!==void 0&&(ve=ve[ye],Array.isArray(ve)&&(ve=ve[0])),ve)if(U>=0&&U<=M.width-L&&z>=0&&z<=M.height-O){he.bindFramebuffer(I.FRAMEBUFFER,ve);let Me=M.textures[ce],Fe=Me.format,Ze=Me.type;M.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+ce);let Se=u0(Me);if(Se.__formatReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(Se.__typeReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let Ke=I.createBuffer();I.bindBuffer(I.PIXEL_PACK_BUFFER,Ke),I.bufferData(I.PIXEL_PACK_BUFFER,fe.byteLength,I.STREAM_READ),I.readPixels(U,z,L,O,se.convert(Fe),se.convert(Ze),0),I.bindBuffer(I.PIXEL_PACK_BUFFER,null);let Vt=te!==null?Re.get(te).__webglFramebuffer:null;he.bindFramebuffer(I.FRAMEBUFFER,Vt);let yt=I.fenceSync(I.SYNC_GPU_COMMANDS_COMPLETE,0);return I.flush(),await LM(I,yt,4),I.bindBuffer(I.PIXEL_PACK_BUFFER,Ke),I.getBufferSubData(I.PIXEL_PACK_BUFFER,0,fe),I.bindBuffer(I.PIXEL_PACK_BUFFER,null),I.deleteBuffer(Ke),I.deleteSync(yt),fe}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(M,U=null,z=0){let L=Math.pow(2,-z),O=Math.floor(M.image.width*L),fe=Math.floor(M.image.height*L),ye=U!==null?U.x:0,ce=U!==null?U.y:0;Le.setTexture2D(M,0),I.copyTexSubImage2D(I.TEXTURE_2D,z,0,0,ye,ce,O,fe),he.unbindTexture()},this.copyTextureToTexture=function(M,U,z=null,L=null,O=0,fe=0){let ye,ce,ve,Me,Fe,Ze,Se,Ke,Vt,yt=M.isCompressedTexture?M.mipmaps[fe]:M.image;if(z!==null)ye=z.max.x-z.min.x,ce=z.max.y-z.min.y,ve=z.isBox3?z.max.z-z.min.z:1,Me=z.min.x,Fe=z.min.y,Ze=z.isBox3?z.min.z:0;else{let tt=Math.pow(2,-O);ye=Math.floor(yt.width*tt),ce=Math.floor(yt.height*tt),M.isDataArrayTexture?ve=yt.depth:M.isData3DTexture?ve=Math.floor(yt.depth*tt):ve=1,Me=0,Fe=0,Ze=0}L!==null?(Se=L.x,Ke=L.y,Vt=L.z):(Se=0,Ke=0,Vt=0);let gt=se.convert(U.format),An=se.convert(U.type),Ae;U.isData3DTexture?(Le.setTexture3D(U,0),Ae=I.TEXTURE_3D):U.isDataArrayTexture||U.isCompressedArrayTexture?(Le.setTexture2DArray(U,0),Ae=I.TEXTURE_2D_ARRAY):(Le.setTexture2D(U,0),Ae=I.TEXTURE_2D),he.activeTexture(I.TEXTURE0),he.pixelStorei(I.UNPACK_FLIP_Y_WEBGL,U.flipY),he.pixelStorei(I.UNPACK_PREMULTIPLY_ALPHA_WEBGL,U.premultiplyAlpha),he.pixelStorei(I.UNPACK_ALIGNMENT,U.unpackAlignment);let Un=he.getParameter(I.UNPACK_ROW_LENGTH),rt=he.getParameter(I.UNPACK_IMAGE_HEIGHT),Vr=he.getParameter(I.UNPACK_SKIP_PIXELS),Mn=he.getParameter(I.UNPACK_SKIP_ROWS),Hc=he.getParameter(I.UNPACK_SKIP_IMAGES);he.pixelStorei(I.UNPACK_ROW_LENGTH,yt.width),he.pixelStorei(I.UNPACK_IMAGE_HEIGHT,yt.height),he.pixelStorei(I.UNPACK_SKIP_PIXELS,Me),he.pixelStorei(I.UNPACK_SKIP_ROWS,Fe),he.pixelStorei(I.UNPACK_SKIP_IMAGES,Ze);let Qs=M.isDataArrayTexture||M.isData3DTexture,kr=U.isDataArrayTexture||U.isData3DTexture;if(M.isDepthTexture){let tt=Re.get(M),pn=Re.get(U),Ui=Re.get(tt.__renderTarget),Bt=Re.get(pn.__renderTarget);he.bindFramebuffer(I.READ_FRAMEBUFFER,Ui.__webglFramebuffer),he.bindFramebuffer(I.DRAW_FRAMEBUFFER,Bt.__webglFramebuffer);for(let rn=0;rn<ve;rn++)Qs&&(I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,Re.get(M).__webglTexture,O,Ze+rn),I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,Re.get(U).__webglTexture,fe,Vt+rn)),I.blitFramebuffer(Me,Fe,Me+ye,Fe+ce,Se,Ke,Se+ye,Ke+ce,I.DEPTH_BUFFER_BIT,I.NEAREST);he.bindFramebuffer(I.READ_FRAMEBUFFER,null),he.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else if(O!==0||M.isRenderTargetTexture||Re.has(M)){let tt=Re.get(M),pn=Re.get(U);he.bindFramebuffer(I.READ_FRAMEBUFFER,D),he.bindFramebuffer(I.DRAW_FRAMEBUFFER,V);for(let Ui=0;Ui<ve;Ui++)Qs?I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,tt.__webglTexture,O,Ze+Ui):I.framebufferTexture2D(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,tt.__webglTexture,O),kr?I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,pn.__webglTexture,fe,Vt+Ui):I.framebufferTexture2D(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,pn.__webglTexture,fe),O!==0?I.blitFramebuffer(Me,Fe,Me+ye,Fe+ce,Se,Ke,Se+ye,Ke+ce,I.COLOR_BUFFER_BIT,I.NEAREST):kr?I.copyTexSubImage3D(Ae,fe,Se,Ke,Vt+Ui,Me,Fe,ye,ce):I.copyTexSubImage2D(Ae,fe,Se,Ke,Me,Fe,ye,ce);he.bindFramebuffer(I.READ_FRAMEBUFFER,null),he.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else kr?M.isDataTexture||M.isData3DTexture?I.texSubImage3D(Ae,fe,Se,Ke,Vt,ye,ce,ve,gt,An,yt.data):U.isCompressedArrayTexture?I.compressedTexSubImage3D(Ae,fe,Se,Ke,Vt,ye,ce,ve,gt,yt.data):I.texSubImage3D(Ae,fe,Se,Ke,Vt,ye,ce,ve,gt,An,yt):M.isDataTexture?I.texSubImage2D(I.TEXTURE_2D,fe,Se,Ke,ye,ce,gt,An,yt.data):M.isCompressedTexture?I.compressedTexSubImage2D(I.TEXTURE_2D,fe,Se,Ke,yt.width,yt.height,gt,yt.data):I.texSubImage2D(I.TEXTURE_2D,fe,Se,Ke,ye,ce,gt,An,yt);he.pixelStorei(I.UNPACK_ROW_LENGTH,Un),he.pixelStorei(I.UNPACK_IMAGE_HEIGHT,rt),he.pixelStorei(I.UNPACK_SKIP_PIXELS,Vr),he.pixelStorei(I.UNPACK_SKIP_ROWS,Mn),he.pixelStorei(I.UNPACK_SKIP_IMAGES,Hc),fe===0&&U.generateMipmaps===!0&&U.mipmapsAutoUpdate===!0&&I.generateMipmap(Ae),he.unbindTexture()},this.initRenderTarget=function(M){Re.get(M).__webglFramebuffer===void 0&&Le.setupRenderTarget(M)},this.initTexture=function(M){M.isCubeTexture?Le.setTextureCube(M,0):M.isData3DTexture?Le.setTexture3D(M,0):M.isDataArrayTexture||M.isCompressedArrayTexture?Le.setTexture2DArray(M,0):Le.setTexture2D(M,0),he.unbindTexture()},this.resetState=function(){q=0,H=0,te=null,he.reset(),ae.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Ci}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;let n=this.getContext();n.drawingBufferColorSpace=je._getDrawingBufferColorSpace(e),n.unpackColorSpace=je._getUnpackColorSpace()}};var F3=(()=>{let t=new Float32Array([-1,-1,0,3,-1,0,-1,3,0]),e=new Float32Array([0,0,2,0,0,2]),n=new ps;return n.setAttribute("position",new li(t,3)),n.setAttribute("uv",new li(e,2)),n})(),Hr=class Jv{static get fullscreenGeometry(){return F3}constructor(e="Pass",n=new Br,i=new Fr){this.name=e,this.renderer=null,this.scene=n,this.camera=i,this.screen=null,this.rtt=!0,this.needsSwap=!0,this.needsDepthBlit=!1,this.needsDepthTexture=!1,this.enabled=!0}get renderToScreen(){return!this.rtt}set renderToScreen(e){if(this.rtt===e){let n=this.fullscreenMaterial;n!==null&&(n.needsUpdate=!0),this.rtt=!e}}set mainScene(e){}set mainCamera(e){}setRenderer(e){this.renderer=e}isEnabled(){return this.enabled}setEnabled(e){this.enabled=e}get fullscreenMaterial(){return this.screen!==null?this.screen.material:null}set fullscreenMaterial(e){let n=this.screen;n!==null?n.material=e:(n=new Qn(Jv.fullscreenGeometry,e),n.frustumCulled=!1,this.scene===null&&(this.scene=new Br),this.scene.add(n),this.screen=n)}getFullscreenMaterial(){return this.fullscreenMaterial}setFullscreenMaterial(e){this.fullscreenMaterial=e}getDepthTexture(){return null}setDepthTexture(e,n=vs){}render(e,n,i,s,r){throw new Error("Render method not implemented!")}setSize(e,n){}initialize(e,n,i){}dispose(){for(let e of Object.keys(this)){let n=this[e];(n instanceof zt||n instanceof Vi||n instanceof qt||n instanceof Jv)&&this[e].dispose()}this.fullscreenMaterial!==null&&this.fullscreenMaterial.dispose()}},z3=class extends Hr{constructor(){super("ClearMaskPass",null,null),this.needsSwap=!1}render(t,e,n,i,s){let r=t.state.buffers.stencil;r.setLocked(!1),r.setTest(!1)}},H3=`#ifdef COLOR_WRITE
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
}`,G3="varying vec2 vUv;void main(){vUv=position.xy*0.5+0.5;gl_Position=vec4(position.xy,1.0,1.0);}",V3=class extends hn{constructor(){super({name:"CopyMaterial",defines:{COLOR_SPACE_CONVERSION:"1",DEPTH_PACKING:"0",COLOR_WRITE:"1"},uniforms:{inputBuffer:new Ut(null),depthBuffer:new Ut(null),channelWeights:new Ut(null),opacity:new Ut(1)},blending:ki,toneMapped:!1,depthWrite:!1,depthTest:!1,fragmentShader:H3,vertexShader:G3}),this.depthFunc=Go}get inputBuffer(){return this.uniforms.inputBuffer.value}set inputBuffer(t){let e=t!==null;this.colorWrite!==e&&(e?this.defines.COLOR_WRITE=!0:delete this.defines.COLOR_WRITE,this.colorWrite=e,this.needsUpdate=!0),this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){let e=t!==null;this.depthWrite!==e&&(e?this.defines.DEPTH_WRITE=!0:delete this.defines.DEPTH_WRITE,this.depthTest=e,this.depthWrite=e,this.needsUpdate=!0),this.uniforms.depthBuffer.value=t}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}get colorSpaceConversion(){return this.defines.COLOR_SPACE_CONVERSION!==void 0}set colorSpaceConversion(t){this.colorSpaceConversion!==t&&(t?this.defines.COLOR_SPACE_CONVERSION=!0:delete this.defines.COLOR_SPACE_CONVERSION,this.needsUpdate=!0)}get channelWeights(){return this.uniforms.channelWeights.value}set channelWeights(t){t!==null?(this.defines.USE_WEIGHTS="1",this.uniforms.channelWeights.value=t):delete this.defines.USE_WEIGHTS,this.needsUpdate=!0}setInputBuffer(t){this.uniforms.inputBuffer.value=t}getOpacity(t){return this.uniforms.opacity.value}setOpacity(t){this.uniforms.opacity.value=t}},k3=class extends Hr{constructor(t,e=!0){super("CopyPass"),this.fullscreenMaterial=new V3,this.needsSwap=!1,this.renderTarget=t,t===void 0&&(this.renderTarget=new zt(1,1,{minFilter:Dt,magFilter:Dt,stencilBuffer:!1,depthBuffer:!1}),this.renderTarget.texture.name="CopyPass.Target"),this.autoResize=e}get resize(){return this.autoResize}set resize(t){this.autoResize=t}get texture(){return this.renderTarget.texture}getTexture(){return this.renderTarget.texture}setAutoResizeEnabled(t){this.autoResize=t}render(t,e,n,i,s){this.fullscreenMaterial.inputBuffer=e.texture,t.setRenderTarget(this.renderToScreen?null:this.renderTarget),t.render(this.scene,this.camera)}setSize(t,e){this.autoResize&&this.renderTarget.setSize(t,e)}initialize(t,e,n){n!==void 0&&(this.renderTarget.texture.type=n,n!==Gt?this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1":t!==null&&t.outputColorSpace===Tt&&(this.renderTarget.texture.colorSpace=Tt))}},_1=new Ye,M1=class extends Hr{constructor(t=!0,e=!0,n=!1){super("ClearPass",null,null),this.needsSwap=!1,this.color=t,this.depth=e,this.stencil=n,this.overrideClearColor=null,this.overrideClearAlpha=-1}setClearFlags(t,e,n){this.color=t,this.depth=e,this.stencil=n}getOverrideClearColor(){return this.overrideClearColor}setOverrideClearColor(t){this.overrideClearColor=t}getOverrideClearAlpha(){return this.overrideClearAlpha}setOverrideClearAlpha(t){this.overrideClearAlpha=t}render(t,e,n,i,s){let r=this.overrideClearColor,a=this.overrideClearAlpha,o=t.getClearAlpha(),l=r!==null,c=a>=0;l?(t.getClearColor(_1),t.setClearColor(r,c?a:o)):c&&t.setClearAlpha(a),t.setRenderTarget(this.renderToScreen?null:e),t.clear(this.color,this.depth,this.stencil),l?t.setClearColor(_1,o):c&&t.setClearAlpha(o)}},W3=class extends Hr{constructor(t,e){super("MaskPass",t,e),this.needsSwap=!1,this.clearPass=new M1(!1,!1,!0),this.inverse=!1}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get inverted(){return this.inverse}set inverted(t){this.inverse=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getClearPass(){return this.clearPass}isInverted(){return this.inverted}setInverted(t){this.inverted=t}render(t,e,n,i,s){let r=t.getContext(),a=t.state.buffers,o=this.scene,l=this.camera,c=this.clearPass,h=this.inverted?0:1,p=1-h;a.color.setMask(!1),a.depth.setMask(!1),a.color.setLocked(!0),a.depth.setLocked(!0),a.stencil.setTest(!0),a.stencil.setOp(r.REPLACE,r.REPLACE,r.REPLACE),a.stencil.setFunc(r.ALWAYS,h,4294967295),a.stencil.setClear(p),a.stencil.setLocked(!0),this.clearPass.enabled&&(this.renderToScreen?c.render(t,null):(c.render(t,e),c.render(t,n))),this.renderToScreen?(t.setRenderTarget(null),t.render(o,l)):(t.setRenderTarget(e),t.render(o,l),t.setRenderTarget(n),t.render(o,l)),a.color.setLocked(!1),a.depth.setLocked(!1),a.stencil.setLocked(!1),a.stencil.setFunc(r.EQUAL,1,4294967295),a.stencil.setOp(r.KEEP,r.KEEP,r.KEEP),a.stencil.setLocked(!0)}};function X3(t,e){let n=t.getContext();if(e<=0||typeof n.renderbufferStorageMultisample!="function")return 0;let i=n.getParameter(n.MAX_SAMPLES),s=Math.min(e,i);if(s<=0)return 0;let r=n.getParameter(n.RENDERBUFFER_BINDING),a=n.createRenderbuffer();try{return n.bindRenderbuffer(n.RENDERBUFFER,a),n.renderbufferStorageMultisample(n.RENDERBUFFER,s,n.RGBA8,1,1),s}catch{return 0}finally{n.bindRenderbuffer(n.RENDERBUFFER,r),n.deleteRenderbuffer(a)}}var Qv=1/1e3,Y3=1e3,q3=class{constructor(){this.startTime=performance.now(),this.previousTime=0,this.currentTime=0,this._delta=0,this._elapsed=0,this._fixedDelta=1e3/60,this.timescale=1,this.useFixedDelta=!1,this._autoReset=!1}get autoReset(){return this._autoReset}set autoReset(t){typeof document<"u"&&document.hidden!==void 0&&(t?document.addEventListener("visibilitychange",this):document.removeEventListener("visibilitychange",this),this._autoReset=t)}get delta(){return this._delta*Qv}get fixedDelta(){return this._fixedDelta*Qv}set fixedDelta(t){this._fixedDelta=t*Y3}get elapsed(){return this._elapsed*Qv}update(t){this.useFixedDelta?this._delta=this.fixedDelta:(this.previousTime=this.currentTime,this.currentTime=(t!==void 0?t:performance.now())-this.startTime,this._delta=this.currentTime-this.previousTime),this._delta*=this.timescale,this._elapsed+=this._delta}reset(){this._delta=0,this._elapsed=0,this.currentTime=performance.now()-this.startTime}getDelta(){return this.delta}getElapsed(){return this.elapsed}handleEvent(t){document.hidden||(this.currentTime=performance.now()-this.startTime)}dispose(){this.autoReset=!1}},E1=class{constructor(t=null,{depthBuffer:e=!0,stencilBuffer:n=!1,multisampling:i=0,frameBufferType:s=Gt}={}){this.renderer=null,this.inputBuffer=this.createBuffer(e,n,s,i),this.outputBuffer=this.inputBuffer.clone(),this.copyPass=new k3,this.depthRenderTarget=null,this.passes=[],this.timer=new q3,this.autoRenderToScreen=!0,this.setRenderer(t)}get stableDepthTexture(){return this.depthRenderTarget===null?null:this.depthRenderTarget.depthTexture}get multisampling(){return this.inputBuffer.samples}set multisampling(t){let e=this.renderer===null?t:X3(this.renderer,t);this.multisampling!==e&&(this.inputBuffer.samples=e,this.outputBuffer.samples=e,this.inputBuffer.dispose(),this.outputBuffer.dispose())}getTimer(){return this.timer}getRenderer(){return this.renderer}setRenderer(t){if(this.renderer=t,t!==null){let e=t.getSize(new Oe),n=t.getContext().getContextAttributes().alpha,i=this.inputBuffer.texture.type;i===Gt&&t.outputColorSpace===Tt&&(this.inputBuffer.texture.colorSpace=Tt,this.outputBuffer.texture.colorSpace=Tt,this.inputBuffer.dispose(),this.outputBuffer.dispose());let s=this.multisampling;this.multisampling=s,t.autoClear=!1,this.setSize(e.width,e.height);for(let r of this.passes)r.initialize(t,n,i)}}replaceRenderer(t,e=!0){let n=this.renderer,i=n.domElement.parentNode;return this.setRenderer(t),e&&i!==null&&(i.removeChild(n.domElement),i.appendChild(t.domElement)),n}createDepthTexture(){let t=new ci;t.name="EffectComposer.InputDepth",this.inputBuffer.stencilBuffer?(t.format=qi,t.type=Ys):t.type=fi;let e=new ci;e.format=t.format,e.type=t.type,e.name="EffectComposer.OutputDepth";let n=new ci;n.format=t.format,n.type=t.type,n.name="EffectComposer.StableDepth",this.inputBuffer.depthTexture=t,this.outputBuffer.depthTexture=e,this.inputBuffer.dispose(),this.outputBuffer.dispose();let{width:i,height:s}=this.inputBuffer;this.depthRenderTarget=new zt(i,s,{depthBuffer:!0,stencilBuffer:this.inputBuffer.stencilBuffer,depthTexture:n})}blitDepthBuffer(t){let e=this.renderer,n=this.depthRenderTarget,i=e.properties,s=e.getContext();e.setRenderTarget(n);let r=i.get(t).__webglFramebuffer,a=i.get(n).__webglFramebuffer,o=t.stencilBuffer?s.DEPTH_BUFFER_BIT|s.STENCIL_BUFFER_BIT:s.DEPTH_BUFFER_BIT;s.bindFramebuffer(s.READ_FRAMEBUFFER,r),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,a),s.blitFramebuffer(0,0,t.width,t.height,0,0,n.width,n.height,o,s.NEAREST),s.bindFramebuffer(s.READ_FRAMEBUFFER,null),s.bindFramebuffer(s.DRAW_FRAMEBUFFER,null),e.setRenderTarget(null)}deleteDepthTexture(){let t=this.stableDepthTexture;for(let e of this.passes)e.getDepthTexture()===t&&e.setDepthTexture(null);this.depthRenderTarget!==null&&(this.depthRenderTarget.dispose(),this.depthRenderTarget=null),this.inputBuffer.depthTexture!==null&&(this.inputBuffer.depthTexture.dispose(),this.inputBuffer.depthTexture=null),this.outputBuffer.depthTexture!==null&&(this.outputBuffer.depthTexture.dispose(),this.outputBuffer.depthTexture=null)}createBuffer(t,e,n,i){let s=this.renderer,r=s===null?new Oe:s.getDrawingBufferSize(new Oe),a=new zt(r.width,r.height,{minFilter:Dt,magFilter:Dt,samples:i,stencilBuffer:e,depthBuffer:t,type:n});return n===Gt&&s!==null&&s.outputColorSpace===Tt&&(a.texture.colorSpace=Tt),a.texture.name="EffectComposer.Buffer",a.texture.generateMipmaps=!1,a}setMainScene(t){for(let e of this.passes)e.mainScene=t}setMainCamera(t){for(let e of this.passes)e.mainCamera=t}addPass(t,e){let n=this.passes,i=this.renderer,s=i.getDrawingBufferSize(new Oe),r=i.getContext().getContextAttributes().alpha,a=this.inputBuffer.texture.type;if(t.renderer=i,t.setSize(s.width,s.height),t.initialize(i,r,a),this.autoRenderToScreen&&(n.length>0&&(n[n.length-1].renderToScreen=!1),t.renderToScreen&&(this.autoRenderToScreen=!1)),e!==void 0?n.splice(e,0,t):n.push(t),this.autoRenderToScreen&&(n[n.length-1].renderToScreen=!0),t.needsDepthTexture||this.depthRenderTarget!==null)if(this.depthRenderTarget===null){this.createDepthTexture();for(let o of n)o.setDepthTexture(this.stableDepthTexture)}else t.setDepthTexture(this.stableDepthTexture)}removePass(t){let e=this.passes,n=e.indexOf(t);if(n!==-1&&e.splice(n,1).length>0){let r=this.stableDepthTexture;if(r!==null){let a=(l,c)=>l||c.needsDepthTexture;e.reduce(a,!1)||(t.getDepthTexture()===r&&t.setDepthTexture(null),this.deleteDepthTexture())}this.autoRenderToScreen&&n===e.length&&(t.renderToScreen=!1,e.length>0&&(e[e.length-1].renderToScreen=!0))}}removeAllPasses(){let t=this.passes;this.deleteDepthTexture(),t.length>0&&(this.autoRenderToScreen&&(t[t.length-1].renderToScreen=!1),this.passes=[])}render(t){let e=this.renderer,n=this.copyPass,i=this.inputBuffer,s=this.outputBuffer,r,a=!1;t===void 0&&(this.timer.update(),t=this.timer.getDelta());for(let o of this.passes)if(o.enabled){if(o.render(e,i,s,t,a),o.needsDepthBlit&&this.depthRenderTarget!==null&&this.blitDepthBuffer(i),o.needsSwap){if(a){n.renderToScreen=o.renderToScreen;let l=e.getContext(),c=e.state.buffers.stencil;c.setFunc(l.NOTEQUAL,1,4294967295),n.render(e,i,s,t,a),c.setFunc(l.EQUAL,1,4294967295)}r=i,i=s,s=r}o instanceof W3?a=!0:o instanceof z3&&(a=!1)}}setSize(t,e,n){let i=this.renderer,s=i.getSize(new Oe);(t===void 0||e===void 0)&&(t=s.width,e=s.height),(s.width!==t||s.height!==e)&&i.setSize(t,e,n);let r=i.getDrawingBufferSize(new Oe);this.inputBuffer.setSize(r.width,r.height),this.outputBuffer.setSize(r.width,r.height),this.depthRenderTarget!==null&&this.depthRenderTarget.setSize(r.width,r.height);for(let a of this.passes)a.setSize(r.width,r.height)}reset(){this.dispose(),this.autoRenderToScreen=!0}dispose(){for(let t of this.passes)t.dispose();this.deleteDepthTexture(),this.inputBuffer.dispose(),this.outputBuffer.dispose(),this.copyPass.dispose(),this.timer.dispose(),this.passes=[],Hr.fullscreenGeometry.dispose()}},wa={NONE:0,DEPTH:1,CONVOLUTION:2},it={FRAGMENT_HEAD:"FRAGMENT_HEAD",FRAGMENT_MAIN_UV:"FRAGMENT_MAIN_UV",FRAGMENT_MAIN_IMAGE:"FRAGMENT_MAIN_IMAGE",VERTEX_HEAD:"VERTEX_HEAD",VERTEX_MAIN_SUPPORT:"VERTEX_MAIN_SUPPORT"},Q3=class{constructor(){this.shaderParts=new Map([[it.FRAGMENT_HEAD,null],[it.FRAGMENT_MAIN_UV,null],[it.FRAGMENT_MAIN_IMAGE,null],[it.VERTEX_HEAD,null],[it.VERTEX_MAIN_SUPPORT,null]]),this.defines=new Map,this.uniforms=new Map,this.blendModes=new Map,this.extensions=new Set,this.attributes=wa.NONE,this.varyings=new Set,this.uvTransformation=!1,this.readDepth=!1,this.colorSpace=Ws}};var Zv=!1,S1=class{constructor(t=null){this.originalMaterials=new Map,this.material=null,this.materials=null,this.materialsBackSide=null,this.materialsDoubleSide=null,this.materialsFlatShaded=null,this.materialsFlatShadedBackSide=null,this.materialsFlatShadedDoubleSide=null,this.setMaterial(t),this.meshCount=0,this.replaceMaterial=e=>{if(e.isMesh){let n;if(e.material.flatShading)switch(e.material.side){case Fn:n=this.materialsFlatShadedDoubleSide;break;case Qt:n=this.materialsFlatShadedBackSide;break;default:n=this.materialsFlatShaded;break}else switch(e.material.side){case Fn:n=this.materialsDoubleSide;break;case Qt:n=this.materialsBackSide;break;default:n=this.materials;break}this.originalMaterials.set(e,e.material),e.isSkinnedMesh?e.material=n[2]:e.isInstancedMesh?e.material=n[1]:e.material=n[0],++this.meshCount}}}cloneMaterial(t){if(!(t instanceof hn))return t.clone();let e=t.uniforms,n=new Map;for(let s in e){let r=e[s].value;r.isRenderTargetTexture&&(e[s].value=null,n.set(s,r))}let i=t.clone();for(let s of n)e[s[0]].value=s[1],i.uniforms[s[0]].value=s[1];return i}setMaterial(t){if(this.disposeMaterials(),this.material=t,t!==null){let e=this.materials=[this.cloneMaterial(t),this.cloneMaterial(t),this.cloneMaterial(t)];for(let n of e)n.uniforms=Object.assign({},t.uniforms),n.side=ms;e[2].skinning=!0,this.materialsBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=Qt,i}),this.materialsDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.side=Fn,i}),this.materialsFlatShaded=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i}),this.materialsFlatShadedBackSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=Qt,i}),this.materialsFlatShadedDoubleSide=e.map(n=>{let i=this.cloneMaterial(n);return i.uniforms=Object.assign({},t.uniforms),i.flatShading=!0,i.side=Fn,i})}}render(t,e,n){let i=t.shadowMap.enabled;if(t.shadowMap.enabled=!1,Zv){let s=this.originalMaterials;this.meshCount=0,e.traverse(this.replaceMaterial),t.render(e,n);for(let r of s)r[0].material=r[1];this.meshCount!==s.size&&s.clear()}else{let s=e.overrideMaterial;e.overrideMaterial=this.material,t.render(e,n),e.overrideMaterial=s}t.shadowMap.enabled=i}disposeMaterials(){if(this.material!==null){let t=this.materials.concat(this.materialsBackSide).concat(this.materialsDoubleSide).concat(this.materialsFlatShaded).concat(this.materialsFlatShadedBackSide).concat(this.materialsFlatShadedDoubleSide);for(let e of t)e.dispose()}}dispose(){this.originalMaterials.clear(),this.disposeMaterials()}static get workaroundEnabled(){return Zv}static set workaroundEnabled(t){Zv=t}};var et={SKIP:9,SET:30,ADD:0,ALPHA:1,AVERAGE:2,COLOR:3,COLOR_BURN:4,COLOR_DODGE:5,DARKEN:6,DIFFERENCE:7,DIVIDE:8,DST:9,EXCLUSION:10,HARD_LIGHT:11,HARD_MIX:12,HUE:13,INVERT:14,INVERT_RGB:15,LIGHTEN:16,LINEAR_BURN:17,LINEAR_DODGE:18,LINEAR_LIGHT:19,LUMINOSITY:20,MULTIPLY:21,NEGATION:22,NORMAL:23,OVERLAY:24,PIN_LIGHT:25,REFLECT:26,SATURATION:27,SCREEN:28,SOFT_LIGHT:29,SRC:30,SUBTRACT:31,VIVID_LIGHT:32},Z3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",K3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,src.a*opacity);}",J3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=(dst.rgb+src.rgb)*0.5;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",j3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.xy,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",$3="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=mix(step(0.0,b)*(1.0-min(vec3(1.0),(1.0-a)/max(b,1e-9))),vec3(1.0),step(1.0,a));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",eU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=dst.rgb,b=src.rgb;vec3 c=step(0.0,a)*mix(min(vec3(1.0),a/max(1.0-b,1e-9)),vec3(1.0),step(1.0,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",tU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",nU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=abs(dst.rgb-src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",iU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb/max(src.rgb,1e-9);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",sU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-2.0*dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",rU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb,1.0);vec3 b=min(src.rgb,1.0);vec3 c=mix(2.0*a*b,1.0-2.0*(1.0-a)*(1.0-b),step(0.5,b));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",aU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=step(1.0,dst.rgb+src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",oU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(b.x,a.yz));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",lU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",cU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=src.rgb*max(1.0-dst.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",uU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb,src.rgb);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",fU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",hU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=min(dst.rgb+src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",dU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=clamp(2.0*src.rgb+dst.rgb-1.0,0.0,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",pU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.xy,b.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",mU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb*src.rgb;return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",gU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(1.0-abs(1.0-dst.rgb-src.rgb),0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",vU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return mix(dst,src,opacity);}",xU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=2.0*src.rgb*dst.rgb;vec3 b=1.0-2.0*(1.0-src.rgb)*(1.0-dst.rgb);vec3 c=mix(a,b,step(0.5,dst.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",yU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 c=mix(mix(src2,dst.rgb,step(0.5*dst.rgb,src.rgb)),max(src2-1.0,vec3(0.0)),step(dst.rgb,src2-1.0));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",_U="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=min(dst.rgb*dst.rgb/max(1.0-src.rgb,1e-9),1.0);vec3 c=mix(a,src.rgb,step(1.0,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",SU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 a=RGBToHSL(dst.rgb);vec3 b=RGBToHSL(src.rgb);vec3 c=HSLToRGB(vec3(a.x,b.y,a.z));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",AU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=dst.rgb+src.rgb-min(dst.rgb*src.rgb,1.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",MU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 src2=2.0*src.rgb;vec3 d=dst.rgb+(src2-1.0);vec3 w=step(0.5,src.rgb);vec3 a=dst.rgb-(1.0-src2)*dst.rgb*(1.0-dst.rgb);vec3 b=mix(d*(sqrt(dst.rgb)-dst.rgb),d*dst.rgb*((16.0*dst.rgb-12.0)*dst.rgb+3.0),w*(1.0-step(0.25,dst.rgb)));vec3 c=mix(a,b,w);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",EU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){return src;}",TU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=max(dst.rgb-src.rgb,0.0);return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",bU="vec4 blend(const in vec4 dst,const in vec4 src,const in float opacity){vec3 c=mix(max(1.0-min((1.0-dst.rgb)/(2.0*src.rgb),1.0),0.0),min(dst.rgb/(2.0*(1.0-src.rgb)),1.0),step(0.5,src.rgb));return mix(dst,vec4(c,max(dst.a,src.a)),opacity);}",wU=new Map([[et.ADD,Z3],[et.ALPHA,K3],[et.AVERAGE,J3],[et.COLOR,j3],[et.COLOR_BURN,$3],[et.COLOR_DODGE,eU],[et.DARKEN,tU],[et.DIFFERENCE,nU],[et.DIVIDE,iU],[et.DST,null],[et.EXCLUSION,sU],[et.HARD_LIGHT,rU],[et.HARD_MIX,aU],[et.HUE,oU],[et.INVERT,lU],[et.INVERT_RGB,cU],[et.LIGHTEN,uU],[et.LINEAR_BURN,fU],[et.LINEAR_DODGE,hU],[et.LINEAR_LIGHT,dU],[et.LUMINOSITY,pU],[et.MULTIPLY,mU],[et.NEGATION,gU],[et.NORMAL,vU],[et.OVERLAY,xU],[et.PIN_LIGHT,yU],[et.REFLECT,_U],[et.SATURATION,SU],[et.SCREEN,AU],[et.SOFT_LIGHT,MU],[et.SRC,EU],[et.SUBTRACT,TU],[et.VIVID_LIGHT,bU]]),CU=class extends qn{constructor(t,e=1){super(),this._blendFunction=t,this.opacity=new Ut(e)}getOpacity(){return this.opacity.value}setOpacity(t){this.opacity.value=t}get blendFunction(){return this._blendFunction}set blendFunction(t){this._blendFunction=t,this.dispatchEvent({type:"change"})}getBlendFunction(){return this.blendFunction}setBlendFunction(t){this.blendFunction=t}getShaderCode(){return wU.get(this.blendFunction)}};var T1=class extends qn{constructor(t,e,{attributes:n=wa.NONE,blendFunction:i=et.NORMAL,defines:s=new Map,uniforms:r=new Map,extensions:a=null,vertexShader:o=null}={}){super(),this.name=t,this.renderer=null,this.attributes=n,this.fragmentShader=e,this.vertexShader=o,this.defines=s,this.uniforms=r,this.extensions=a,this.blendMode=new CU(i),this.blendMode.addEventListener("change",l=>this.setChanged()),this._inputColorSpace=Ws,this._outputColorSpace=hi}get inputColorSpace(){return this._inputColorSpace}set inputColorSpace(t){this._inputColorSpace=t,this.setChanged()}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t,this.setChanged()}set mainScene(t){}set mainCamera(t){}getName(){return this.name}setRenderer(t){this.renderer=t}getDefines(){return this.defines}getUniforms(){return this.uniforms}getExtensions(){return this.extensions}getBlendMode(){return this.blendMode}getAttributes(){return this.attributes}setAttributes(t){this.attributes=t,this.setChanged()}getFragmentShader(){return this.fragmentShader}setFragmentShader(t){this.fragmentShader=t,this.setChanged()}getVertexShader(){return this.vertexShader}setVertexShader(t){this.vertexShader=t,this.setChanged()}setChanged(){this.dispatchEvent({type:"change"})}setDepthTexture(t,e=vs){}update(t,e,n){}setSize(t,e){}initialize(t,e,n){}dispose(){for(let t of Object.keys(this)){let e=this[t];(e instanceof zt||e instanceof Vi||e instanceof qt||e instanceof Hr)&&this[t].dispose()}}};var qL=[new Float32Array([0,0]),new Float32Array([0,1,1]),new Float32Array([0,1,1,2]),new Float32Array([0,1,2,2,3]),new Float32Array([0,1,2,3,4,4,5]),new Float32Array([0,1,2,3,4,5,7,8,9,10])];var b1=class extends Hr{constructor(t,e,n=null){super("RenderPass",t,e),this.needsSwap=!1,this.needsDepthBlit=!0,this.clearPass=new M1,this.overrideMaterialManager=n===null?null:new S1(n),this.ignoreBackground=!1,this.skipShadowMapUpdate=!1,this.selection=null}set mainScene(t){this.scene=t}set mainCamera(t){this.camera=t}get renderToScreen(){return super.renderToScreen}set renderToScreen(t){super.renderToScreen=t,this.clearPass.renderToScreen=t}get overrideMaterial(){let t=this.overrideMaterialManager;return t!==null?t.material:null}set overrideMaterial(t){let e=this.overrideMaterialManager;t!==null?e!==null?e.setMaterial(t):this.overrideMaterialManager=new S1(t):e!==null&&(e.dispose(),this.overrideMaterialManager=null)}getOverrideMaterial(){return this.overrideMaterial}setOverrideMaterial(t){this.overrideMaterial=t}get clear(){return this.clearPass.enabled}set clear(t){this.clearPass.enabled=t}getSelection(){return this.selection}setSelection(t){this.selection=t}isBackgroundDisabled(){return this.ignoreBackground}setBackgroundDisabled(t){this.ignoreBackground=t}isShadowMapDisabled(){return this.skipShadowMapUpdate}setShadowMapDisabled(t){this.skipShadowMapUpdate=t}getClearPass(){return this.clearPass}render(t,e,n,i,s){let r=this.scene,a=this.camera,o=this.selection,l=a.layers.mask,c=r.background,h=t.shadowMap.autoUpdate,p=this.renderToScreen?null:e;o!==null&&a.layers.set(o.getLayer()),this.skipShadowMapUpdate&&(t.shadowMap.autoUpdate=!1),(this.ignoreBackground||this.clearPass.overrideClearColor!==null)&&(r.background=null),this.clearPass.enabled&&this.clearPass.render(t,e),t.setRenderTarget(p),this.overrideMaterialManager!==null?this.overrideMaterialManager.render(t,r,a):t.render(r,a),a.layers.mask=l,r.background=c,t.shadowMap.autoUpdate=h}};var QL=Math.PI*.5;var RU=`#include <common>
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
}`,DU="uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;VERTEX_HEAD void main(){vUv=position.xy*0.5+0.5;VERTEX_MAIN_SUPPORT gl_Position=vec4(position.xy,1.0,1.0);}",UU=class extends hn{constructor(t,e,n,i,s=!1){super({name:"EffectMaterial",defines:{THREE_REVISION:"187".replace(/\D+/g,""),DEPTH_PACKING:"0",ENCODE_OUTPUT:"1"},uniforms:{inputBuffer:new Ut(null),depthBuffer:new Ut(null),resolution:new Ut(new Oe),texelSize:new Ut(new Oe),cameraNear:new Ut(.3),cameraFar:new Ut(1e3),aspect:new Ut(1),time:new Ut(0)},blending:ki,toneMapped:!1,depthWrite:!1,depthTest:!1,dithering:s}),t&&this.setShaderParts(t),e&&this.setDefines(e),n&&this.setUniforms(n),this.copyCameraSettings(i)}set inputBuffer(t){this.uniforms.inputBuffer.value=t}setInputBuffer(t){this.uniforms.inputBuffer.value=t}get depthBuffer(){return this.uniforms.depthBuffer.value}set depthBuffer(t){this.uniforms.depthBuffer.value=t}get depthPacking(){return Number(this.defines.DEPTH_PACKING)}set depthPacking(t){this.defines.DEPTH_PACKING=t.toFixed(0),this.needsUpdate=!0}setDepthBuffer(t,e=vs){this.depthBuffer=t,this.depthPacking=e}setShaderData(t){this.setShaderParts(t.shaderParts),this.setDefines(t.defines),this.setUniforms(t.uniforms),this.setExtensions(t.extensions)}setShaderParts(t){return this.fragmentShader=RU.replace(it.FRAGMENT_HEAD,t.get(it.FRAGMENT_HEAD)||"").replace(it.FRAGMENT_MAIN_UV,t.get(it.FRAGMENT_MAIN_UV)||"").replace(it.FRAGMENT_MAIN_IMAGE,t.get(it.FRAGMENT_MAIN_IMAGE)||""),this.vertexShader=DU.replace(it.VERTEX_HEAD,t.get(it.VERTEX_HEAD)||"").replace(it.VERTEX_MAIN_SUPPORT,t.get(it.VERTEX_MAIN_SUPPORT)||""),this.needsUpdate=!0,this}setDefines(t){for(let e of t.entries())this.defines[e[0]]=e[1];return this.needsUpdate=!0,this}setUniforms(t){for(let e of t.entries())this.uniforms[e[0]]=e[1];return this}setExtensions(t){this.extensions={};for(let e of t)this.extensions[e]=!0;return this}get encodeOutput(){return this.defines.ENCODE_OUTPUT!==void 0}set encodeOutput(t){this.encodeOutput!==t&&(t?this.defines.ENCODE_OUTPUT="1":delete this.defines.ENCODE_OUTPUT,this.needsUpdate=!0)}isOutputEncodingEnabled(t){return this.encodeOutput}setOutputEncodingEnabled(t){this.encodeOutput=t}get time(){return this.uniforms.time.value}set time(t){this.uniforms.time.value=t}setDeltaTime(t){this.uniforms.time.value+=t}adoptCameraSettings(t){this.copyCameraSettings(t)}copyCameraSettings(t){t&&(this.uniforms.cameraNear.value=t.near,this.uniforms.cameraFar.value=t.far,t instanceof On?this.defines.PERSPECTIVE_CAMERA="1":delete this.defines.PERSPECTIVE_CAMERA,this.needsUpdate=!0)}setSize(t,e){let n=this.uniforms;n.resolution.value.set(t,e),n.texelSize.value.set(1/t,1/e),n.aspect.value=t/e}static get Section(){return it}};var KL=Number("187".replace(/\D+/g,"")),ba=255/256,JL=new Float32Array([ba/256**3,ba/256**2,ba/256,ba]),jL=new Float32Array([ba,ba/256,ba/256**2,1/256**3]);function A1(t,e,n){for(let i of e){let s="$1"+t+i.charAt(0).toUpperCase()+i.slice(1),r=new RegExp("([^\\.])(\\b"+i+"\\b)","g");for(let a of n.entries())a[1]!==null&&n.set(a[0],a[1].replace(r,s))}}function IU(t,e,n){let i=e.getFragmentShader(),s=e.getVertexShader(),r=i!==void 0&&/mainImage/.test(i),a=i!==void 0&&/mainUv/.test(i);if(n.attributes|=e.getAttributes(),i===void 0)throw new Error(`Missing fragment shader (${e.name})`);if(a&&(n.attributes&wa.CONVOLUTION)!==0)throw new Error(`Effects that transform UVs are incompatible with convolution effects (${e.name})`);if(!r&&!a)throw new Error(`Could not find mainImage or mainUv function (${e.name})`);{let o=/\w+\s+(\w+)\([\w\s,]*\)\s*{/g,l=n.shaderParts,c=l.get(it.FRAGMENT_HEAD)||"",h=l.get(it.FRAGMENT_MAIN_UV)||"",p=l.get(it.FRAGMENT_MAIN_IMAGE)||"",u=l.get(it.VERTEX_HEAD)||"",d=l.get(it.VERTEX_MAIN_SUPPORT)||"",x=new Set,A=new Set;if(a&&(h+=`	${t}MainUv(UV);
`,n.uvTransformation=!0),s!==null&&/mainSupport/.test(s)){let m=/mainSupport *\([\w\s]*?uv\s*?\)/.test(s);d+=`	${t}MainSupport(`,d+=m?`vUv);
`:`);
`;for(let y of s.matchAll(/(?:varying\s+\w+\s+([\S\s]*?);)/g))for(let _ of y[1].split(/\s*,\s*/))n.varyings.add(_),x.add(_),A.add(_);for(let y of s.matchAll(o))A.add(y[1])}for(let m of i.matchAll(o))A.add(m[1]);for(let m of e.defines.keys())A.add(m.replace(/\([\w\s,]*\)/g,""));for(let m of e.uniforms.keys())A.add(m);A.delete("while"),A.delete("for"),A.delete("if"),e.uniforms.forEach((m,y)=>n.uniforms.set(t+y.charAt(0).toUpperCase()+y.slice(1),m)),e.defines.forEach((m,y)=>n.defines.set(t+y.charAt(0).toUpperCase()+y.slice(1),m));let g=new Map([["fragment",i],["vertex",s]]);A1(t,A,n.defines),A1(t,A,g),i=g.get("fragment"),s=g.get("vertex");let f=e.blendMode;if(n.blendModes.set(f.blendFunction,f),r){e.inputColorSpace!==null&&e.inputColorSpace!==n.colorSpace&&(p+=e.inputColorSpace===Tt?`color0 = sRGBTransferOETF(color0);
	`:`color0 = sRGBToLinear(color0);
	`),e.outputColorSpace!==hi?n.colorSpace=e.outputColorSpace:e.inputColorSpace!==null&&(n.colorSpace=e.inputColorSpace);let m=/MainImage *\([\w\s,]*?depth[\w\s,]*?\)/;p+=`${t}MainImage(color0, UV, `,(n.attributes&wa.DEPTH)!==0&&m.test(i)&&(p+="depth, ",n.readDepth=!0),p+=`color1);
	`;let y=t+"BlendOpacity";n.uniforms.set(y,f.opacity),p+=`color0 = blend${f.blendFunction}(color0, color1, ${y});

	`,c+=`uniform float ${y};

`}if(c+=i+`
`,s!==null&&(u+=s+`
`),l.set(it.FRAGMENT_HEAD,c),l.set(it.FRAGMENT_MAIN_UV,h),l.set(it.FRAGMENT_MAIN_IMAGE,p),l.set(it.VERTEX_HEAD,u),l.set(it.VERTEX_MAIN_SUPPORT,d),e.extensions!==null)for(let m of e.extensions)n.extensions.add(m)}}var w1=class extends Hr{constructor(t,...e){super("EffectPass"),this.fullscreenMaterial=new UU(null,null,null,t),this.listener=n=>this.handleEvent(n),this.effects=[],this.setEffects(e),this.skipRendering=!1,this.minTime=1,this.maxTime=Number.POSITIVE_INFINITY,this.timeScale=1}set mainScene(t){for(let e of this.effects)e.mainScene=t}set mainCamera(t){this.fullscreenMaterial.copyCameraSettings(t);for(let e of this.effects)e.mainCamera=t}get encodeOutput(){return this.fullscreenMaterial.encodeOutput}set encodeOutput(t){this.fullscreenMaterial.encodeOutput=t}get dithering(){return this.fullscreenMaterial.dithering}set dithering(t){let e=this.fullscreenMaterial;e.dithering=t,e.needsUpdate=!0}setEffects(t){for(let e of this.effects)e.removeEventListener("change",this.listener);this.effects=t.sort((e,n)=>n.attributes-e.attributes);for(let e of this.effects)e.addEventListener("change",this.listener)}updateMaterial(){let t=new Q3,e=0;for(let a of this.effects)if(a.blendMode.blendFunction===et.DST)t.attributes|=a.getAttributes()&wa.DEPTH;else{if((t.attributes&a.getAttributes()&wa.CONVOLUTION)!==0)throw new Error(`Convolution effects cannot be merged (${a.name})`);IU("e"+e++,a,t)}let n=t.shaderParts.get(it.FRAGMENT_HEAD),i=t.shaderParts.get(it.FRAGMENT_MAIN_IMAGE),s=t.shaderParts.get(it.FRAGMENT_MAIN_UV),r=/\bblend\b/g;for(let a of t.blendModes.values())n+=a.getShaderCode().replace(r,`blend${a.blendFunction}`)+`
`;(t.attributes&wa.DEPTH)!==0?(t.readDepth&&(i=`float depth = readDepth(UV);

	`+i),this.needsDepthTexture=this.getDepthTexture()===null):this.needsDepthTexture=!1,t.colorSpace===Tt&&(i+=`color0 = sRGBToLinear(color0);
	`),t.uvTransformation?(s=`vec2 transformedUv = vUv;
`+s,t.defines.set("UV","transformedUv")):t.defines.set("UV","vUv"),t.shaderParts.set(it.FRAGMENT_HEAD,n),t.shaderParts.set(it.FRAGMENT_MAIN_IMAGE,i),t.shaderParts.set(it.FRAGMENT_MAIN_UV,s);for(let[a,o]of t.shaderParts)o!==null&&t.shaderParts.set(a,o.trim().replace(/^#/,`
#`));this.skipRendering=e===0,this.needsSwap=!this.skipRendering,this.fullscreenMaterial.setShaderData(t)}recompile(){this.updateMaterial()}getDepthTexture(){return this.fullscreenMaterial.depthBuffer}setDepthTexture(t,e=vs){this.fullscreenMaterial.depthBuffer=t,this.fullscreenMaterial.depthPacking=e;for(let n of this.effects)n.setDepthTexture(t,e)}render(t,e,n,i,s){for(let r of this.effects)r.update(t,e,i);if(!this.skipRendering||this.renderToScreen){let r=this.fullscreenMaterial;r.inputBuffer=e.texture,r.time+=i*this.timeScale,t.setRenderTarget(this.renderToScreen?null:n),t.render(this.scene,this.camera)}}setSize(t,e){this.fullscreenMaterial.setSize(t,e);for(let n of this.effects)n.setSize(t,e)}initialize(t,e,n){this.renderer=t;for(let i of this.effects)i.initialize(t,e,n);this.updateMaterial(),n!==void 0&&n!==Gt&&(this.fullscreenMaterial.defines.FRAMEBUFFER_PRECISION_HIGH="1")}dispose(){super.dispose();for(let t of this.effects)t.removeEventListener("change",this.listener),t.dispose()}handleEvent(t){t.type==="change"&&this.recompile()}};var eP=[new Float32Array(3),new Float32Array(3)],tP=[new Float32Array(3),new Float32Array(3),new Float32Array(3),new Float32Array(3)],nP=[[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([0,1,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([0,1,1]),new Float32Array([1,1,1])]];var iP=[new Float32Array(2),new Float32Array(2)];var sP=new Float32Array([0,-.25,.25,-.125,.125,-.375,.375]),rP=[new Float32Array([0,0]),new Float32Array([.25,-.25]),new Float32Array([-.25,.25]),new Float32Array([.125,-.125]),new Float32Array([-.125,.125])],aP=[new Uint8Array([0,0]),new Uint8Array([3,0]),new Uint8Array([0,3]),new Uint8Array([3,3]),new Uint8Array([1,0]),new Uint8Array([4,0]),new Uint8Array([1,3]),new Uint8Array([4,3]),new Uint8Array([0,1]),new Uint8Array([3,1]),new Uint8Array([0,4]),new Uint8Array([3,4]),new Uint8Array([1,1]),new Uint8Array([4,1]),new Uint8Array([1,4]),new Uint8Array([4,4])],oP=[new Uint8Array([0,0]),new Uint8Array([1,0]),new Uint8Array([0,2]),new Uint8Array([1,2]),new Uint8Array([2,0]),new Uint8Array([3,0]),new Uint8Array([2,2]),new Uint8Array([3,2]),new Uint8Array([0,1]),new Uint8Array([1,1]),new Uint8Array([0,3]),new Uint8Array([1,3]),new Uint8Array([2,1]),new Uint8Array([3,1]),new Uint8Array([2,3]),new Uint8Array([3,3])];var lP=new Map([[Dn(0,0,0,0),new Float32Array([0,0,0,0])],[Dn(0,0,0,1),new Float32Array([0,0,0,1])],[Dn(0,0,1,0),new Float32Array([0,0,1,0])],[Dn(0,0,1,1),new Float32Array([0,0,1,1])],[Dn(0,1,0,0),new Float32Array([0,1,0,0])],[Dn(0,1,0,1),new Float32Array([0,1,0,1])],[Dn(0,1,1,0),new Float32Array([0,1,1,0])],[Dn(0,1,1,1),new Float32Array([0,1,1,1])],[Dn(1,0,0,0),new Float32Array([1,0,0,0])],[Dn(1,0,0,1),new Float32Array([1,0,0,1])],[Dn(1,0,1,0),new Float32Array([1,0,1,0])],[Dn(1,0,1,1),new Float32Array([1,0,1,1])],[Dn(1,1,0,0),new Float32Array([1,1,0,0])],[Dn(1,1,0,1),new Float32Array([1,1,0,1])],[Dn(1,1,1,0),new Float32Array([1,1,1,0])],[Dn(1,1,1,1),new Float32Array([1,1,1,1])]]);function Kv(t,e,n){return t+(e-t)*n}function Dn(t,e,n,i){let s=Kv(t,e,.75),r=Kv(n,i,1-.25);return Kv(s,r,1-.125)}var Sd=Wr(nl());var U1=Wr(Fc()),LU=()=>{let e=document.createElement("canvas");e.width=64,e.height=64;let n=e.getContext("2d");if(!n)throw new Error("2D context not available");n.fillStyle="black",n.fillRect(0,0,e.width,e.height);let i=new qt(e);i.minFilter=Dt,i.magFilter=Dt,i.generateMipmaps=!1;let s=[],r=null,a=64,o=.1*64,l=1/a,c=()=>{n.fillStyle="black",n.fillRect(0,0,e.width,e.height)},h=d=>{let x={x:d.x*64,y:(1-d.y)*64},A=1,g=_=>Math.sin(_*Math.PI/2),f=_=>-_*(_-2);d.age<a*.3?A=g(d.age/(a*.3)):A=f(1-(d.age-a*.3)/(a*.7))||0,A*=d.force;let m=`${(d.vx+1)/2*255}, ${(d.vy+1)/2*255}, ${A*255}`,y=320;n.shadowOffsetX=y,n.shadowOffsetY=y,n.shadowBlur=o,n.shadowColor=`rgba(${m},${.22*A})`,n.beginPath(),n.fillStyle="rgba(255,0,0,1)",n.arc(x.x-y,x.y-y,o,0,Math.PI*2),n.fill()};return{texture:i,addTouch:d=>{let x=0,A=0,g=0;if(r){let f=d.x-r.x,m=d.y-r.y;if(f===0&&m===0)return;let y=f*f+m*m,_=Math.sqrt(y);A=f/(_||1),g=m/(_||1),x=Math.min(y*1e4,1)}r={x:d.x,y:d.y},s.push({x:d.x,y:d.y,age:0,force:x,vx:A,vy:g})},update:()=>{c();for(let d=s.length-1;d>=0;d-=1){let x=s[d],A=x.force*l*(1-x.age/a);x.x+=x.vx*A,x.y+=x.vy*A,x.age+=1,x.age>a&&s.splice(d,1)}s.forEach(h),i.needsUpdate=!0},set radiusScale(d){o=.1*64*d}}},PU=(t,e)=>{let n=`
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
  `;return new T1("LiquidEffect",n,{uniforms:new Map([["uTexture",new Ut(t)],["uStrength",new Ut(e?.strength??.025)],["uTime",new Ut(0)],["uFreq",new Ut(e?.freq??4.5)]])})},OU={square:0,circle:1,triangle:2,diamond:3},FU=`
void main() {
  gl_Position = vec4(position, 1.0);
}
`,zU=`
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
`,jv=10;function $v({variant:t="square",pixelSize:e=4,color:n="#B497CF",className:i="",style:s,antialias:r=!0,patternScale:a=2,patternDensity:o=1,liquid:l=!1,liquidStrength:c=.1,liquidRadius:h=1,pixelSizeJitter:p=0,enableRipples:u=!0,rippleIntensityScale:d=1,rippleThickness:x=.1,rippleSpeed:A=.3,liquidWobbleSpeed:g=4.5,autoPauseOffscreen:f=!0,speed:m=.5,transparent:y=!0,edgeFade:_=.5}){let T=(0,Sd.useRef)(null);return(0,Sd.useEffect)(()=>{let b=T.current;if(!b)return;let v=document.createElement("canvas"),E=new gd({canvas:v,antialias:r,alpha:!0,powerPreference:"high-performance"});E.domElement.style.width="100%",E.domElement.style.height="100%",E.setPixelRatio(Math.min(window.devicePixelRatio||1,2)),b.appendChild(E.domElement),y?E.setClearAlpha(0):E.setClearColor(0,1);let N={uResolution:{value:new Oe(0,0)},uTime:{value:0},uColor:{value:new Ye(n)},uClickPos:{value:Array.from({length:jv},()=>new Oe(-1,-1))},uClickTimes:{value:new Float32Array(jv)},uShapeType:{value:OU[t]??0},uPixelSize:{value:e*E.getPixelRatio()},uScale:{value:a},uDensity:{value:o},uPixelJitter:{value:p},uEnableRipples:{value:u?1:0},uRippleSpeed:{value:A},uRippleThickness:{value:x},uRippleIntensity:{value:d},uEdgeFade:{value:_}},C=new Br,B=new Fr(-1,1,1,-1,0,1),F=new hn({vertexShader:FU,fragmentShader:zU,uniforms:N,transparent:!0,depthTest:!1,depthWrite:!1,glslVersion:Ic}),X=new Qn(new Aa(2,2),F);C.add(X);let D,V,q;if(l){V=LU(),V.radiusScale=h,D=new E1(E),D.addPass(new b1(C,B)),q=PU(V.texture,{strength:c,freq:g});let be=new w1(B,q);be.renderToScreen=!0,D.addPass(be)}let H=()=>{let be=b.clientWidth||1,qe=b.clientHeight||1;E.setSize(be,qe,!1),N.uResolution.value.set(E.domElement.width,E.domElement.height),N.uPixelSize.value=e*E.getPixelRatio(),D?.setSize(E.domElement.width,E.domElement.height)},te=new ResizeObserver(H);te.observe(b),H();let k=be=>{let qe=E.domElement.getBoundingClientRect(),Ce=E.domElement.width/qe.width,st=E.domElement.height/qe.height;return{x:(be.clientX-qe.left)*Ce,y:(qe.height-(be.clientY-qe.top))*st,width:E.domElement.width,height:E.domElement.height}},$=0,ee=be=>{let qe=k(be);N.uClickPos.value[$].set(qe.x,qe.y),N.uClickTimes.value[$]=N.uTime.value,$=($+1)%jv},_e=be=>{if(!V)return;let qe=k(be);V.addTouch({x:qe.x/qe.width,y:qe.y/qe.height})};window.addEventListener("pointerdown",ee,{passive:!0}),window.addEventListener("pointermove",_e,{passive:!0});let Ie=!0,Q=()=>{Ie=!document.hidden};document.addEventListener("visibilitychange",Q);let de=new Mc,xe=Math.random()*1e3,Te=0,pe=()=>{Te=requestAnimationFrame(pe),!(f&&!Ie)&&(N.uTime.value=xe+de.getElapsedTime()*m,q&&(q.uniforms.get("uTime").value=N.uTime.value),D?(V?.update(),D.render()):E.render(C,B))};return Te=requestAnimationFrame(pe),()=>{cancelAnimationFrame(Te),te.disconnect(),window.removeEventListener("pointerdown",ee),window.removeEventListener("pointermove",_e),document.removeEventListener("visibilitychange",Q),X.geometry.dispose(),F.dispose(),D?.dispose(),E.dispose(),E.forceContextLoss(),E.domElement.parentElement===b&&b.removeChild(E.domElement)}},[r,f,n,_,u,l,h,c,g,o,a,e,p,d,A,x,m,y,t]),(0,U1.jsx)("div",{ref:T,className:`pixel-blast-container ${i}`,style:s,"aria-hidden":"true"})}var e0=Wr(Fc());function HU(){let[t,e]=(0,Ad.useState)(document.documentElement.dataset.theme==="light");return(0,Ad.useEffect)(()=>{let n=new MutationObserver(()=>{e(document.documentElement.dataset.theme==="light")});return n.observe(document.documentElement,{attributes:!0,attributeFilter:["data-theme"]}),()=>n.disconnect()},[]),(0,e0.jsx)($v,{variant:"square",pixelSize:4,color:t?"#000000":"#ffffff",patternScale:2,patternDensity:1.4,pixelSizeJitter:.5,enableRipples:!0,rippleSpeed:.4,rippleThickness:.12,rippleIntensityScale:1.5,liquid:!1,speed:.5,edgeFade:0,transparent:!0})}var I1=document.getElementById("pixel-blast-root");I1&&(0,B1.createRoot)(I1).render((0,e0.jsx)(HU,{}));})();
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
   * postprocessing v6.39.6 build Sun Nov 01 2026
   * https://github.com/pmndrs/postprocessing
   * Copyright 2015-2026 Raoul van Rüschen
   * @license Zlib
   *)
*/
